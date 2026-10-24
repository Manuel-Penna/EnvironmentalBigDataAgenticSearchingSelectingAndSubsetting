from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import fsspec
import numpy as np
import xarray as xr
import zarr

from metadata_dataclasses import *

class BaseScientificExtractor(ABC):
    def extract(self, source: str) -> DatasetMetadata:
        warnings: list[str] = []
        
        is_remote = self._is_remote_source(source)
        
        # If the file is local, calculate its size
        size_bytes = None
        if not is_remote and os.path.exists(source):
            try:
                size_bytes = os.path.getsize(source)
            except Exception as e:
                warnings.append(f"Cannot calculate file size: {e}")

        # Use private method to open dataset 
        ds = self._open_dataset(source)

        dataset_id = self._compute_dataset_id(source)
        global_attributes = self._safe_convert_dict(ds.attrs)

        dimensions = {
            str(dim): int(size)
            for dim, size in ds.sizes.items()
        }

        variables = self._extract_variables(ds, warnings)
        temporal = self._extract_temporal_metadata(ds, warnings)
        spatial = self._extract_spatial_metadata(ds, warnings)

        return DatasetMetadata(
            dataset_id=dataset_id,
            source=source,
            format=self.dataset_format(),
            is_remote=is_remote,
            size_bytes=size_bytes,
            global_attributes=global_attributes,
            dimensions=dimensions,
            spatial=spatial,
            temporal=temporal,
            variables=variables,
            warnings=warnings
        )

    # ABSTRACT METHODS
    @abstractmethod
    def _open_dataset(self, source: str) -> xr.Dataset:
        pass

    @abstractmethod
    def dataset_format(self) -> str:
        pass

    # VARIABLE EXTRACTION
    def _extract_variables(
        self, ds: xr.Dataset, warnings: list[str]
    ) -> list[VariableMetadata]:
        variables = []

        for var_name, var in ds.data_vars.items():
            try:
                chunks = var.chunks
                if chunks is not None:
                    chunks = [list(chunk_group) for chunk_group in chunks]

                statistics = self._compute_statistics(var, var_name, warnings)

                variable_metadata = VariableMetadata(
                    name=str(var_name),
                    dtype=str(var.dtype),
                    dimensions=list(var.dims),
                    shape=list(var.shape),
                    attributes=self._safe_convert_dict(var.attrs),
                    chunks=chunks,
                    statistics=statistics,
                    long_name=str(var.attrs.get("long_name") or var.attrs.get("standard_name")),
                )
                variables.append(variable_metadata)

            except Exception as e:
                warnings.append(f"Failed extracting variable '{var_name}': {e}")

        return variables

    
    def _compute_statistics(
        self,
        var: xr.DataArray,
        var_name: str,
        warnings: list[str],
    ) -> VariableStatistics | None:
        """
        Compute basic descriptive statistics for a variable.

        Only runs on numeric dtypes. Uses numpy nanfunctions so that
        _FillValue / NaN values are excluded from all computations.
        Loads the data into memory — for very large remote files this
        may be slow; a warning is appended if computation fails.
        """

        """
        if not np.issubdtype(var.dtype, np.number):
            return None

        total_elements = 1
        for s in var.shape:
            total_elements *= s

        MAX_STAT_ELEMENTS = 50_000_000   # ~200 MB for float32
        if total_elements > MAX_STAT_ELEMENTS:
            warnings.append(
                f"Variable '{var_name}' too large for statistics "
                f"({total_elements:,} elements) — skipped."
            )
            return None
        """

        # Skip non-numeric variables (e.g. string coordinates, flags)
        if not np.issubdtype(var.dtype, np.number):
            return None

        try:
            # .values triggers computation if the array is dask-backed
            data = var.values.astype(np.float64)

            # Apply _FillValue / missing_value masking if not already NaN
            fill_value = var.attrs.get("_FillValue") or var.attrs.get("missing_value")
            if fill_value is not None:
                try:
                    data = np.where(data == float(fill_value), np.nan, data)
                except (ValueError, TypeError):
                    pass

            valid   = data[~np.isnan(data)]
            n_total = data.size
            n_valid = valid.size
            n_nan   = n_total - n_valid

            if n_valid == 0:
                return VariableStatistics(
                    count=0, nan_count=n_nan
                )

            return VariableStatistics(
                mean=      float(np.mean(valid)),
                std=       float(np.std(valid)),
                min=       float(np.min(valid)),
                max=       float(np.max(valid)),
                median=    float(np.median(valid)),
                count=     int(n_valid),
                nan_count= int(n_nan),
            )

        except Exception as e:
            warnings.append(
                f"Failed computing statistics for '{var_name}': {e}"
            )
            return None

    # TEMPORAL METADATA
    def _extract_temporal_metadata(self, ds: xr.Dataset, warnings: list[str]) -> TemporalMetadata:
        possible_time_names = ["time", "times", "date", "t"]

        for name in possible_time_names:
            if name in ds.coords:
                try:
                    values = ds[name].values
                    if len(values) == 0:
                        return TemporalMetadata()

                    #safely convert the date (might be numpy datetime64) to string
                    start = str(values.min())
                    end = str(values.max())

                    return TemporalMetadata(start=start, end=end)
                except Exception as e:
                    warnings.append(f"Failed extracting temporal metadata on '{name}': {e}")

        return TemporalMetadata()

    # SPATIAL METADATA
    def _extract_spatial_metadata(self, ds: xr.Dataset, warnings: list[str]) -> SpatialMetadata:
        lat_candidates = ["lat", "latitude", "nav_lat", "rlat", "y"]
        lon_candidates = ["lon", "longitude", "nav_lon", "rlon", "x"]

        lat_name = next((c for c in lat_candidates if c in ds.coords), None)
        lon_name = next((c for c in lon_candidates if c in ds.coords), None)

        if not lat_name or not lon_name:
            return SpatialMetadata()

        try:
            lat_values = ds[lat_name].values
            lon_values = ds[lon_name].values

            return SpatialMetadata(
                lat_min=float(np.nanmin(lat_values)),  # nanmin avoids crashes if NaN values are present
                lat_max=float(np.nanmax(lat_values)),
                lon_min=float(np.nanmin(lon_values)),
                lon_max=float(np.nanmax(lon_values))
            )
        except Exception as e:
            warnings.append(f"Failed extracting spatial metadata: {e}")
            return SpatialMetadata()

    # REMOTE SOURCE VALIDATION
    def _is_remote_source(self, source: str) -> bool:
        parsed = urlparse(source)
        # No scheme (es. C:\file o /home/user/file) means local file
        if not parsed.scheme:
            return False
        return parsed.scheme in ["http", "https", "ftp", "s3", "gcs"]

    # DATASET ID
    def _compute_dataset_id(self, source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    # SAFE CONVERSION
    def _safe_convert_dict(self, obj: dict[str, Any]) -> dict[str, Any]:
        converted = {}
        for key, value in obj.items():
            converted[str(key)] = self._safe_convert_value(value)
        return converted

    def _safe_convert_value(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return [self._safe_convert_value(v) for v in value]
        if isinstance(value, dict):
            return self._safe_convert_dict(value)
        return value


# NETCDF EXTRACTOR
class NetCDFExtractor(BaseScientificExtractor):

    def dataset_format(self) -> str:
        return "netcdf"

    def _open_dataset(self, source: str) -> xr.Dataset:
        if self._is_remote_source(source):
            try:
                return xr.open_dataset(source, engine="netcdf4", decode_times=True, chunks={})
            except Exception:
                # Normalize to dap2:// to suppress PyDAP protocol warning
                dap_url = source.replace("http://", "dap2://", 1).replace("https://", "dap4://", 1)
                return xr.open_dataset(dap_url, engine="pydap", decode_times=True, chunks={})

        # Local file — try engines in order of format coverage
        engines = [
            "h5netcdf",   # NetCDF4 / HDF5
            "netcdf4",    # NetCDF4 / NetCDF3 (via libnetcdf)
            "scipy",      # NetCDF3 classic (no C library needed)
        ]

        last_exc = None
        for engine in engines:
            try:
                return xr.open_dataset(source, engine=engine, decode_times=True, chunks={})
            except Exception as e:
                last_exc = e

        raise RuntimeError(f"Cannot open NetCDF dataset '{source}': {last_exc}")


# ZARR EXTRACTOR
class ZarrExtractor(BaseScientificExtractor):

    def dataset_format(self) -> str:
        return "zarr"

    def _open_dataset(self, source: str) -> xr.Dataset:
        try:
            # Attempt 1: standard scientific Zarr (with metadata that allow us to use xarray to open the file)
            if self._is_remote_source(source):
                mapper = fsspec.get_mapper(source)
                return xr.open_zarr(mapper, consolidated=True, chunks={})
            return xr.open_zarr(source, consolidated=True, chunks={})

        except Exception:
            try:
                # Attempt 2: consolidated=False
                if self._is_remote_source(source):
                    mapper = fsspec.get_mapper(source)
                    return xr.open_zarr(mapper, consolidated=False, chunks={})
                return xr.open_zarr(source, consolidated=False, chunks={})

            except Exception:
                # Attempt 3: fallback for OME-Zarr o non-standard Zarr
                #Manually build a xr.Dataset from Zarr groups
                return self._open_as_generic_zarr(source)

    def _open_as_generic_zarr(self, source: str) -> xr.Dataset:
        try:
            if self._is_remote_source(source):
                mapper = fsspec.get_mapper(source)
                root = zarr.open_group(mapper, mode="r")
            else:
                root = zarr.open_group(source, mode="r")

            data_vars = {}

            def _collect_arrays(group: zarr.Group, prefix: str = "") -> None:
                for name, obj in group.members():          
                    full_name = f"{prefix}_{name}" if prefix else name
                    if isinstance(obj, zarr.Array):
                        dims = [f"dim_{i}" for i in range(obj.ndim)]
                        attrs = dict(obj.attrs) if obj.attrs else {}
                        data_vars[full_name] = xr.Variable(dims=dims, data=obj, attrs=attrs)
                    elif isinstance(obj, zarr.Group):
                        _collect_arrays(obj, prefix=full_name)  # recursion for sub-groups

            _collect_arrays(root)

            if not data_vars:
                raise RuntimeError("No array found in the Zarr group.")

            global_attrs = dict(root.attrs) if root.attrs else {}
            return xr.Dataset(data_vars, attrs=global_attrs)

        except Exception as e:
            raise RuntimeError(f"Cannot open Zarr dataset '{source}': {e}")