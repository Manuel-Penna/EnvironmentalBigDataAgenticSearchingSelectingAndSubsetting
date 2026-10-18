from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

def normalize_lon_range(lo: float, hi: float, tol: float = 1.0):
    if lo >= -180 and hi <= 180:
        return lo, hi                     # already normalized
    if hi - lo >= 360 - tol:
        return -180.0, 180.0              # global / near-global grid
    lo = (lo + 180) % 360 - 180
    hi = (hi + 180) % 360 - 180
    if lo > hi:                           # crosses the antimeridian
        return -180.0, 180.0              # conservative: accept false positives
    return lo, hi

@dataclass
class VariableStatistics:
    mean:     float | None = None
    std:      float | None = None
    min:      float | None = None
    max:      float | None = None
    median:   float | None = None
    count:    int   | None = None    # number of non-NaN values
    nan_count: int  | None = None    # number of NaN/fill values

@dataclass
class VariableMetadata:
    name: str
    dtype: str
    dimensions: list[str]
    shape: list[int]
    attributes: dict[str, Any]
    chunks: list[list[int]] | None = None
    statistics: VariableStatistics | None = None


@dataclass
class SpatialMetadata:
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None

    def __post_init__(self):
        if self.lon_min is not None and self.lon_max is not None:
            self.lon_min, self.lon_max = normalize_lon_range(self.lon_min, self.lon_max)

@dataclass
class TemporalMetadata:
    start: str | None = None
    end: str | None = None


@dataclass
class DatasetMetadata:
    dataset_id: str
    source: str
    format: str
    is_remote: bool
    size_bytes: int | None
    global_attributes: dict[str, Any]
    dimensions: dict[str, int]
    spatial: SpatialMetadata
    temporal: TemporalMetadata
    variables: list[VariableMetadata]
    warnings: list[str]