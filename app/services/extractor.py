"""
app/services/extractor.py
-------------------------
Thin wrapper around NetCDFExtractor and ZarrExtractor.
Keeps route handlers free of format-specific logic.
"""

from __future__ import annotations

from metadata_dataclasses import DatasetMetadata
from extractors import NetCDFExtractor, ZarrExtractor

_netcdf_extractor = NetCDFExtractor()
_zarr_extractor   = ZarrExtractor()


def extract_metadata(source: str, fmt: str) -> DatasetMetadata:
    """
    Extract metadata from a NetCDF or Zarr source.

    Args:
        source: Local path or remote URL.
        fmt:    "netcdf" or "zarr".

    Returns:
        DatasetMetadata instance.

    Raises:
        ValueError:   Unknown format.
        RuntimeError: Extraction failed.
    """
    if fmt == "netcdf":
        return _netcdf_extractor.extract(source)
    elif fmt == "zarr":
        return _zarr_extractor.extract(source)
    else:
        raise ValueError(f"Unknown format: '{fmt}'. Must be 'netcdf' or 'zarr'.")
