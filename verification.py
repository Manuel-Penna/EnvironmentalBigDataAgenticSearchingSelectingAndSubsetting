"""
verify_subset: sanity-check a subset AFTER it was written.

Returns
-------
{"ok": bool,                      # False if any issue has severity "error"
 "issues": [{"severity": "error"|"warning", "code": str,
             "variable": str|None, "message": str}],
 "summary": {var: {"min", "max", "mean", "nan_fraction", "units", ...}},
 "n_errors": int, "n_warnings": int}

Checks
------
  structure   no variables, zero-length dimensions
  per variable  empty, all NaN, high NaN fraction, infinite values, constant
                values, missing units, units mismatch
  vs. catalog   values outside the dataset's known min/max (usually leaked fill
                values or a scaling bug) - pass GeoMeta's stored statistics as
                reference_stats
  vs. request   coordinates outside the requested bbox / time range
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

try:
    from .subsetting import (EPS, SubsetError, SubsetRequest, _clean_time,
                             _coerce_request, east_offset, find_axis,
                             open_if_needed)
except ImportError:  # flat layout
    from subsetting import (EPS, SubsetError, SubsetRequest, _clean_time,  # type: ignore
                            _coerce_request, east_offset, find_axis,
                             open_if_needed)


def verify_subset(source: "str | Path | xr.Dataset",
                  request: "SubsetRequest | dict | None" = None,
                  reference_stats: dict[str, dict[str, float]] | None = None,
                  expected_units: dict[str, str] | None = None,
                  max_nan_fraction: float = 0.5) -> dict:
    """
    source           path to the written subset (or an open Dataset)
    request          the SubsetRequest used; enables the coordinate checks
    reference_stats  {"sst": {"min": 271.0, "max": 313.0}} from the catalog
    expected_units   {"sst": "K"} from the catalog
    """
    issues: list[dict] = []
    summary: dict[str, dict] = {}

    def add(severity: str, code: str, message: str, variable: str | None = None):
        issues.append({"severity": severity, "code": code,
                       "variable": variable, "message": message})

    opened = False
    ds = None
    try:
        ds, opened = open_if_needed(source)

        # ---- structure ---------------------------------------------------
        if not ds.data_vars:
            add("error", "no_variables", "Output contains no data variables.")
        for dim, n in ds.sizes.items():
            if n == 0:
                add("error", "empty_dimension", f"Dimension '{dim}' has length 0.")

        # ---- per variable ------------------------------------------------
        for name in ds.data_vars:
            da = ds[name]
            info: dict[str, Any] = {"shape": list(da.shape), "dtype": str(da.dtype),
                                    "units": da.attrs.get("units")}
            summary[name] = info
            if da.size == 0:
                add("error", "empty_variable", "Variable has no values.", name)
                continue
            if not np.issubdtype(da.dtype, np.number):
                info["skipped"] = "non-numeric"
                continue

            da = da.load()
            n = int(da.size)
            valid = int(da.count())
            info["nan_fraction"] = round(1 - valid / n, 6)
            if valid == 0:
                add("error", "all_nan", "Every value is NaN (no data in the "
                    "selected region/period, or only fill values).", name)
                continue
            if info["nan_fraction"] > max_nan_fraction:
                add("warning", "high_nan_fraction",
                    f"{info['nan_fraction']:.0%} of values are NaN.", name)
            if np.issubdtype(da.dtype, np.floating) and bool(np.isinf(da).any()):
                add("error", "infinite_values", "Variable contains +/-inf.", name)

            vmin, vmax, vmean = (float(da.min()), float(da.max()), float(da.mean()))
            info.update(min=vmin, max=vmax, mean=vmean)
            if valid > 1 and vmin == vmax:
                add("warning", "constant_values",
                    f"All valid values equal {vmin}.", name)

            units = da.attrs.get("units")
            if units is None:
                add("warning", "missing_units", "No 'units' attribute.", name)
            if expected_units and name in expected_units and units != expected_units[name]:
                add("error", "units_mismatch",
                    f"units are {units!r}, expected {expected_units[name]!r}.", name)

            ref = (reference_stats or {}).get(name)
            if ref and ref.get("min") is not None and ref.get("max") is not None:
                tol = 1e-4 * max(abs(ref["max"] - ref["min"]), 1e-12)
                if vmin < ref["min"] - tol or vmax > ref["max"] + tol:
                    add("error", "outside_reference_range",
                        f"Values [{vmin:.6g}, {vmax:.6g}] fall outside the "
                        f"dataset's known range [{ref['min']:.6g}, {ref['max']:.6g}]; "
                        "possible fill values or scaling error.", name)

        # ---- coordinates vs. request ------------------------------------
        if request is not None:
            req = _coerce_request(request)
            _check_coordinates(ds, req, add)

    except SubsetError as e:
        add("error", e.code, e.message)
    except Exception as e:  # noqa: BLE001
        add("error", "unexpected", f"{type(e).__name__}: {e}")
    finally:
        if opened and ds is not None:
            ds.close()

    n_err = sum(i["severity"] == "error" for i in issues)
    return {"ok": n_err == 0, "issues": issues, "summary": summary,
            "n_errors": n_err, "n_warnings": len(issues) - n_err}


def _check_coordinates(ds: xr.Dataset, req: SubsetRequest, add) -> None:
    if req.bbox:
        lat_min, lat_max, lon_min, lon_max = req.bbox
        lat_name, lon_name = find_axis(ds, "lat"), find_axis(ds, "lon")
        if lat_name is None or lon_name is None:
            add("warning", "axis_not_found", "Could not find lat/lon to verify the bbox.")
        else:
            lat = np.asarray(ds[lat_name].values, dtype=float)
            if lat.size and (lat.min() < lat_min - EPS or lat.max() > lat_max + EPS):
                add("error", "coords_outside_request",
                    f"Latitudes [{lat.min()}, {lat.max()}] exceed requested "
                    f"[{lat_min}, {lat_max}].")
            span = lon_max - lon_min
            span = span + 360.0 if span < 0 else span
            if span < 360.0 - EPS:
                lon = np.asarray(ds[lon_name].values, dtype=float)
                key = east_offset(lon, lon_min)
                if lon.size and key.max() > span + EPS:
                    add("error", "coords_outside_request",
                        f"Longitudes exceed requested {lon_min}..{lon_max} (eastward).")

    if req.time and any(t is not None for t in req.time):
        tname = find_axis(ds, "time")
        if tname is None or tname not in ds.indexes:
            add("warning", "axis_not_found", "Could not find time axis to verify the range.")
        else:
            index = ds.indexes[tname]
            start, end = (_clean_time(t) for t in req.time)
            try:
                inside = len(np.arange(len(index))[index.slice_indexer(start, end)])
                if inside != len(index):
                    add("error", "coords_outside_request",
                        f"{len(index) - inside} timestamps fall outside "
                        f"[{start}, {end}].")
            except Exception as e:  # noqa: BLE001
                add("warning", "time_check_failed", f"{type(e).__name__}: {e}")
