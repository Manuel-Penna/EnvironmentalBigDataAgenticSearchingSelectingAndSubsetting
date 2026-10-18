"""
Deterministic subsetting tools for GeoMeta (no LLM involved).

Public API
----------
open_source(source)                              -> lazy xr.Dataset
estimate_subset_size(source, request, max_bytes) -> dict   (reads no data)
subset_dataset(source, request, output_path, ...) -> dict  (writes Zarr/NetCDF)

All public functions return plain dicts and never raise for *expected* failures.
Failures look like:
    {"ok": False, "error": {"code": "...", "message": "...", "details": {...}}}
so an agent can read the reason and adapt.

Request conventions
-------------------
bbox = (lat_min, lat_max, lon_min, lon_max), longitudes in degrees east.
  * Any convention is accepted (-180..180 or 0..360).
  * The box runs WEST -> EAST from lon_min to lon_max. If lon_max < lon_min the
    box crosses the antimeridian (170, -170 is a 20 degree wide box).
  * Output longitudes are expressed in the *request's* convention and are
    continuous (e.g. 170..190 for the box above).
time = (start, end), ISO 8601 strings, either side may be None (open-ended).
  * Date-only bounds are inclusive of the whole day: ("2010-01-01", "2020-12-31")
    includes every timestamp on 2020-12-31.
stride = {"lat": 4, "lon": 4, "time": 2} keeps every n-th point (logical axis
  names or real dimension names).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from extractors import NetCDFExtractor, ZarrExtractor

EPS = 1e-6  # degrees; tolerance for grid points sitting exactly on a bbox edge

_TIME_INDEX_TYPES: tuple = (pd.DatetimeIndex,) + (
    (xr.CFTimeIndex,) if hasattr(xr, "CFTimeIndex") else ()
)

# Encoding keys that are safe to carry from the source into the subset.
# Everything else (chunks, compressors, filters, source, ...) describes the
# *source* storage layout and breaks writing a differently shaped subset.
_KEEP_ENCODING = {
    "dtype", "_FillValue", "missing_value", "scale_factor", "add_offset",
    "units", "calendar",
}


# --------------------------------------------------------------------------
# Errors and request model
# --------------------------------------------------------------------------
class SubsetError(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "error": {"code": self.code, "message": self.message,
                      "details": self.details},
        }


@dataclass
class SubsetRequest:
    variables: list[str] | None = None
    bbox: tuple[float, float, float, float] | None = None
    time: tuple[str | None, str | None] | None = None
    stride: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SubsetRequest":
        bbox = d.get("bbox")
        if isinstance(bbox, dict):
            bbox = (bbox["lat_min"], bbox["lat_max"],
                    bbox["lon_min"], bbox["lon_max"])
        time = d.get("time")
        if isinstance(time, dict):
            time = (time.get("start"), time.get("end"))
        return cls(
            variables=list(d["variables"]) if d.get("variables") else None,
            bbox=tuple(float(x) for x in bbox) if bbox else None,
            time=tuple(time) if time else None,
            stride=dict(d.get("stride") or {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce_request(request: "SubsetRequest | dict") -> SubsetRequest:
    return request if isinstance(request, SubsetRequest) else SubsetRequest.from_dict(request)


@dataclass
class SubsetPlan:
    dataset: xr.Dataset                 # lazily subsetted dataset
    selections: dict[str, np.ndarray]   # dim -> selected integer indices
    axes: dict[str, str]                # logical axis -> coordinate name
    warnings: list[str]
    request: SubsetRequest


# --------------------------------------------------------------------------
# Opening
# --------------------------------------------------------------------------
def _guess_format(source: str) -> str:
    """'zarr' for *.zarr paths and local directories, otherwise 'netcdf'.
    A remote Zarr store without a .zarr suffix cannot be guessed: pass the
    format stored in the catalog (source_format="zarr")."""
    if source.rstrip("/").endswith(".zarr"):
        return "zarr"
    if not source.startswith(("http://", "https://", "s3://", "gcs://", "ftp://")) \
            and Path(source).is_dir():
        return "zarr"
    return "netcdf"


def open_source(source: str | Path, fmt: str | None = None) -> xr.Dataset:
    """Open a dataset exactly the way ingestion does (same fallback chains),
    by delegating to the extractors' ``_open_dataset``."""
    s = str(source)
    fmt = fmt or _guess_format(s)
    if fmt not in {"netcdf", "zarr"}:
        raise SubsetError("bad_format", f"Unknown source format '{fmt}'.")
    extractor = ZarrExtractor() if fmt == "zarr" else NetCDFExtractor()
    try:
        return extractor._open_dataset(s)
    except Exception as e:  # noqa: BLE001
        raise SubsetError("open_failed", f"Could not open {s} as {fmt}: {e}") from e


def open_if_needed(source: "str | Path | xr.Dataset",
                   fmt: str | None = None) -> tuple[xr.Dataset, bool]:
    """Return (dataset, opened_here). Datasets passed in are never closed."""
    if isinstance(source, xr.Dataset):
        return source, False
    return open_source(source, fmt), True


# --------------------------------------------------------------------------
# Axis detection
# --------------------------------------------------------------------------
_AXIS_SPECS = {
    "lat": {"names": {"lat", "latitude"}, "standard": {"latitude"},
            "units": {"degrees_north", "degree_north", "degrees_n", "degree_n"}},
    "lon": {"names": {"lon", "longitude"}, "standard": {"longitude"},
            "units": {"degrees_east", "degree_east", "degrees_e", "degree_e"}},
}


def find_axis(ds: xr.Dataset, kind: str, require_1d: bool = True) -> str | None:
    """Find the coordinate name for 'lat', 'lon' or 'time'."""
    names = list(ds.coords) + [n for n in ds.variables if n not in ds.coords]
    for name in names:
        var = ds.variables[name]
        std = str(var.attrs.get("standard_name", "")).lower()
        units = str(var.attrs.get("units", "")).lower()
        lname = str(name).lower()
        if kind == "time":
            hit = (lname in {"time", "valid_time"} or std == "time"
                   or var.attrs.get("axis") == "T"
                   or np.issubdtype(var.dtype, np.datetime64))
        else:
            spec = _AXIS_SPECS[kind]
            hit = (lname in spec["names"] or std in spec["standard"]
                   or units in spec["units"])
        if hit and (var.ndim == 1 or not require_1d):
            return str(name)
    return None


def _require_axis(ds: xr.Dataset, kind: str) -> str:
    name = find_axis(ds, kind)
    if name is not None:
        return name
    if kind != "time" and find_axis(ds, kind, require_1d=False) is not None:
        raise SubsetError(
            "unsupported_grid",
            f"The {kind} coordinate is multi-dimensional (curvilinear or "
            "projected grid). Only regular 1-D lat/lon grids are supported.",
        )
    raise SubsetError(f"no_{kind}_axis", f"No {kind} coordinate found in dataset.",
                      coordinates=[str(c) for c in ds.coords])


# --------------------------------------------------------------------------
# Planning (lazy, reads only 1-D coordinate arrays)
# --------------------------------------------------------------------------
def _clean_time(t: Any) -> str | None:
    if t is None:
        return None
    s = str(t).strip()
    for suffix in ("Z", "+00:00"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def east_offset(lon_vals: np.ndarray, origin: float) -> np.ndarray:
    """Degrees east of `origin` in [0, 360). Points within EPS west of the
    origin are snapped to a tiny negative offset so they count as on the edge."""
    k = np.mod(np.asarray(lon_vals, dtype=float) - origin, 360.0)
    return np.where(k > 360.0 - EPS, k - 360.0, k)


def _as_indexer(idx: np.ndarray):
    """Contiguous / regularly strided index arrays become slices (cheaper)."""
    if idx.size == 1:
        return slice(int(idx[0]), int(idx[0]) + 1)
    if idx.size > 1:
        d = np.diff(idx)
        if d[0] > 0 and np.all(d == d[0]):
            return slice(int(idx[0]), int(idx[-1]) + 1, int(d[0]))
    return idx


def plan_subset(ds: xr.Dataset, request: "SubsetRequest | dict") -> SubsetPlan:
    req = _coerce_request(request)
    warnings: list[str] = []

    names = list(req.variables) if req.variables else list(ds.data_vars)
    if not names:
        raise SubsetError("no_variables", "Dataset has no data variables.")
    missing = [v for v in names if v not in ds.data_vars]
    if missing:
        raise SubsetError("unknown_variable",
                          f"Variables not in dataset: {missing}",
                          missing=missing, available=list(ds.data_vars)[:100])
    sub = ds[names]

    axes: dict[str, str] = {}
    axis_dims: dict[str, str] = {}
    for kind in ("time", "lat", "lon"):
        name = find_axis(ds, kind)
        if name:
            axes[kind] = name
            axis_dims[kind] = ds[name].dims[0]

    selections: dict[str, np.ndarray] = {}
    lon_origin: float | None = None

    # ---- time -----------------------------------------------------------
    if req.time and any(t is not None for t in req.time):
        tname = _require_axis(ds, "time")
        tdim = ds[tname].dims[0]
        if tdim not in sub.dims:
            warnings.append(f"Selected variables have no '{tdim}' dimension; "
                            "time filter ignored.")
        else:
            index = ds.indexes.get(tname)
            if index is None or not isinstance(index, _TIME_INDEX_TYPES):
                raise SubsetError(
                    "time_not_decoded",
                    f"Time coordinate '{tname}' is not decoded to datetimes "
                    f"(dtype={ds[tname].dtype}).")
            if not index.is_monotonic_increasing:
                raise SubsetError("time_not_sorted",
                                  f"Time coordinate '{tname}' is not sorted ascending.")
            start, end = (_clean_time(t) for t in req.time)
            try:
                sl = index.slice_indexer(start, end)
            except Exception as e:  # noqa: BLE001
                raise SubsetError("bad_time", f"Cannot interpret time range "
                                  f"{req.time}: {e}") from e
            idx = np.arange(len(index))[sl]
            if idx.size == 0:
                raise SubsetError(
                    "empty_selection",
                    f"No timestamps in [{start}, {end}]. Dataset spans "
                    f"{index[0]} to {index[-1]}.",
                    axis="time", requested=[start, end],
                    available=[str(index[0]), str(index[-1])])
            selections[tdim] = idx

    # ---- space ----------------------------------------------------------
    if req.bbox:
        lat_min, lat_max, lon_min, lon_max = req.bbox
        if lat_min > lat_max:
            raise SubsetError("bad_bbox", f"lat_min ({lat_min}) > lat_max ({lat_max}).")
        if lat_min < -90 or lat_max > 90:
            raise SubsetError("bad_bbox", "Latitude must be within [-90, 90].")
        lat_name = _require_axis(ds, "lat")
        lon_name = _require_axis(ds, "lon")
        lat_dim, lon_dim = ds[lat_name].dims[0], ds[lon_name].dims[0]
        if lat_dim == lon_dim:
            raise SubsetError("unsupported_grid",
                              "lat and lon share one dimension (station or "
                              "unstructured data); bbox subsetting not supported.")

        if lat_dim in sub.dims:
            lat_vals = np.asarray(ds[lat_name].values, dtype=float)
            idx = np.nonzero((lat_vals >= lat_min - EPS) & (lat_vals <= lat_max + EPS))[0]
            if idx.size == 0:
                raise SubsetError(
                    "empty_selection",
                    f"No grid rows with latitude in [{lat_min}, {lat_max}]. "
                    f"Dataset spans {np.nanmin(lat_vals)} to {np.nanmax(lat_vals)}.",
                    axis="lat", requested=[lat_min, lat_max],
                    available=[float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals))])
            selections[lat_dim] = idx

        span = lon_max - lon_min
        if span < 0:
            span += 360.0
        if lon_dim in sub.dims and span < 360.0 - EPS:
            lon_vals = np.asarray(ds[lon_name].values, dtype=float)
            key = east_offset(lon_vals, lon_min)  # degrees east of lon_min
            idx = np.nonzero(key <= span + EPS)[0]
            if idx.size == 0:
                raise SubsetError(
                    "empty_selection",
                    f"No grid columns between longitude {lon_min} and {lon_max} "
                    f"(eastward). Dataset spans {np.nanmin(lon_vals)} to "
                    f"{np.nanmax(lon_vals)}.",
                    axis="lon", requested=[lon_min, lon_max],
                    available=[float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))])
            idx = idx[np.argsort(key[idx], kind="stable")]  # west -> east
            selections[lon_dim] = idx
            lon_origin = lon_min

    # ---- stride ---------------------------------------------------------
    for key_name, step in req.stride.items():
        step = int(step)
        if step < 1:
            raise SubsetError("bad_stride", f"Stride for '{key_name}' must be >= 1.")
        dim = axis_dims.get(key_name, key_name)
        if dim not in sub.dims:
            raise SubsetError("unknown_dimension",
                              f"Cannot apply stride: no dimension '{key_name}'.",
                              available=list(sub.dims))
        current = selections.get(dim, np.arange(sub.sizes[dim]))
        selections[dim] = current[::step]

    # ---- apply (still lazy) ---------------------------------------------
    if selections:
        sub = sub.isel({d: _as_indexer(i) for d, i in selections.items()})

    if lon_origin is not None:
        lon_name, lon_dim = axes["lon"], axis_dims["lon"]
        orig = np.asarray(ds[lon_name].values, dtype=float)[selections[lon_dim]]
        new = lon_origin + east_offset(orig, lon_origin)
        if not np.allclose(new, orig, atol=1e-9):
            sub = sub.assign_coords({lon_name: (
                lon_dim, new.astype(ds[lon_name].dtype), dict(ds[lon_name].attrs))})
            warnings.append("Longitudes were converted to the request's convention "
                            "and made continuous (values may run past 180 when the "
                            "box crosses the antimeridian).")

    return SubsetPlan(dataset=sub, selections=selections, axes=axes,
                      warnings=warnings, request=req)


# --------------------------------------------------------------------------
# Size estimation
# --------------------------------------------------------------------------
def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _source_chunks(var: xr.DataArray) -> tuple[int, ...] | None:
    enc = var.encoding
    chunks = enc.get("chunksizes") or enc.get("chunks")  # netCDF4 / zarr
    if chunks and len(chunks) == var.ndim:
        return tuple(int(c) for c in chunks)
    return None


def _estimate(src: xr.Dataset, plan: SubsetPlan, max_bytes: int | None) -> dict:
    out = plan.dataset
    per_var: dict[str, dict] = {}
    total_out = total_read = 0
    for name in out.data_vars:
        o, s = out[name], src[name]
        n_out = int(np.prod([out.sizes[d] for d in o.dims], dtype=np.int64)) if o.dims else 1
        out_bytes = n_out * o.dtype.itemsize

        disk_itemsize = np.dtype(s.encoding.get("dtype", s.dtype)).itemsize
        chunks = _source_chunks(s)
        if chunks is None:
            read_bytes = n_out * disk_itemsize  # contiguous / server-side: lower bound
        else:
            n_touched, chunk_elems = 1, 1
            for dim, size, c in zip(s.dims, s.shape, chunks):
                sel = plan.selections.get(dim)
                n_touched *= (-(-size // c)) if sel is None else len(np.unique(sel // c))
                chunk_elems *= c
            full = int(np.prod(s.shape, dtype=np.int64)) * disk_itemsize
            read_bytes = min(n_touched * chunk_elems * disk_itemsize, full)

        per_var[name] = {"shape": [int(out.sizes[d]) for d in o.dims],
                         "dims": list(o.dims), "dtype": str(o.dtype),
                         "output_bytes": out_bytes, "bytes_read": read_bytes,
                         "chunked": chunks is not None}
        total_out += out_bytes
        total_read += read_bytes

    return {
        "ok": True,
        "shape": {str(d): int(n) for d, n in out.sizes.items()},
        "variables": per_var,
        "output_bytes": total_out, "output_human": _human(total_out),
        "bytes_read": total_read, "read_human": _human(total_read),
        "read_amplification": round(total_read / total_out, 2) if total_out else None,
        "max_bytes": max_bytes,
        "within_limit": None if max_bytes is None else total_out <= max_bytes,
        "note": "Uncompressed sizes; bytes_read is chunk-aligned. Upper bounds "
                "for compressed stores.",
        "warnings": list(plan.warnings),
        "request": plan.request.to_dict(),
    }


def estimate_subset_size(source: "str | Path | xr.Dataset",
                         request: "SubsetRequest | dict",
                         max_bytes: int | None = None,
                         source_format: str | None = None) -> dict:
    """Size of the subset BEFORE reading any data. Also rejects empty overlaps."""
    opened = False
    ds = None
    try:
        ds, opened = open_if_needed(source, source_format)
        plan = plan_subset(ds, request)
        return _estimate(ds, plan, max_bytes)
    except SubsetError as e:
        return e.to_dict()
    except Exception as e:  # noqa: BLE001
        return SubsetError("unexpected", f"{type(e).__name__}: {e}").to_dict()
    finally:
        if opened and ds is not None:
            ds.close()


# --------------------------------------------------------------------------
# Subsetting
# --------------------------------------------------------------------------
def _infer_format(path: Path) -> str:
    return "netcdf" if path.suffix.lower() in {".nc", ".nc4", ".netcdf"} else "zarr"


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _prepare_for_write(ds: xr.Dataset, fmt: str, source: Any,
                       req: SubsetRequest) -> xr.Dataset:
    out = ds.copy()  # new Variable objects: editing encoding won't touch the source
    for v in out.variables.values():
        v.encoding = {k: val for k, val in v.encoding.items() if k in _KEEP_ENCODING}
    out.attrs = dict(out.attrs)
    out.attrs["geometa_subset_source"] = (
        "<in-memory dataset>" if isinstance(source, xr.Dataset) else str(source))
    out.attrs["geometa_subset_request"] = json.dumps(req.to_dict())

    if any(v.chunks for v in out.data_vars.values()):
        out = out.chunk("auto")  # fancy indexing can leave irregular dask chunks
    if fmt == "netcdf":
        for name, v in out.data_vars.items():
            if np.issubdtype(v.dtype, np.number):
                v.encoding.update(zlib=True, complevel=4)
    return out


def subset_dataset(source: "str | Path | xr.Dataset",
                   request: "SubsetRequest | dict",
                   output_path: str | Path,
                   fmt: str | None = None,
                   max_bytes: int | None = None,
                   overwrite: bool = False,
                   source_format: str | None = None) -> dict:
    """Select variables / bbox / time (+ optional stride) and write the result.

    fmt: "zarr" or "netcdf" (default: inferred from the output suffix, else zarr).
    max_bytes: refuse (error code "too_large") if the estimated output exceeds it.
    source_format: "netcdf" or "zarr" (the catalog's `format`); guessed if omitted.
    """
    opened = False
    ds = None
    out_path = Path(output_path)
    try:
        ds, opened = open_if_needed(source, source_format)
        plan = plan_subset(ds, request)
        est = _estimate(ds, plan, max_bytes)
        if max_bytes is not None and est["output_bytes"] > max_bytes:
            raise SubsetError(
                "too_large",
                f"Estimated output {est['output_human']} exceeds the limit "
                f"{_human(max_bytes)}. Narrow the region/time, drop variables, "
                "or add a stride.",
                estimate=est)

        fmt = fmt or _infer_format(out_path)
        if fmt not in {"zarr", "netcdf"}:
            raise SubsetError("bad_format", f"Unknown format '{fmt}'.")
        if out_path.exists():
            if not overwrite:
                raise SubsetError("output_exists", f"{out_path} already exists.")
            shutil.rmtree(out_path) if out_path.is_dir() else out_path.unlink()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out = _prepare_for_write(plan.dataset, fmt, source, plan.request)
        try:
            if fmt == "zarr":
                out.to_zarr(out_path, mode="w")
            else:
                try:
                    import netCDF4  # noqa: F401
                    out.to_netcdf(out_path, engine="netcdf4")
                except ImportError:
                    out.to_netcdf(out_path)
        except Exception as e:  # noqa: BLE001
            if out_path.exists():
                shutil.rmtree(out_path) if out_path.is_dir() else out_path.unlink()
            raise SubsetError("write_failed", f"{type(e).__name__}: {e}") from e

        return {
            "ok": True,
            "output_path": str(out_path),
            "format": fmt,
            "shape": est["shape"],
            "variables": list(plan.dataset.data_vars),
            "estimated_output_bytes": est["output_bytes"],
            "bytes_read": est["bytes_read"],
            "file_bytes": _path_bytes(out_path),
            "warnings": plan.warnings,
            "request": plan.request.to_dict(),
            "source": ("<in-memory dataset>" if isinstance(source, xr.Dataset)
                       else str(source)),
        }
    except SubsetError as e:
        return e.to_dict()
    except Exception as e:  # noqa: BLE001
        return SubsetError("unexpected", f"{type(e).__name__}: {e}").to_dict()
    finally:
        if opened and ds is not None:
            ds.close()