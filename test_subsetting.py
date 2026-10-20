"""Run with:  pytest -q test_subsetting.py

Synthetic data: value = t * 1e6 + lat * 1000 + lon (float64), so every output
value can be checked against the exact grid point it must come from.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from subsetting import estimate_subset_size, open_source, subset_dataset
from verification import verify_subset


def make_ds(lat, lon, time, var="sst", units="K"):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    t = np.arange(len(time))[:, None, None]
    data = t * 1e6 + lat[None, :, None] * 1000 + lon[None, None, :]
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data, {"units": units})},
        coords={"time": time, "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})},
    )
    return ds


def to_zarr(ds, path, chunks=None):
    ds = ds.chunk(chunks) if chunks else ds
    ds.to_zarr(path, mode="w")
    return str(path)


DAILY = pd.date_range("2000-01-01", "2001-12-31", freq="D")


# ---------------------------------------------------------------- basics
def test_time_and_space_together(tmp_path):
    src = to_zarr(make_ds(np.arange(25, 51), np.arange(-10, 46), DAILY),
                  tmp_path / "a.zarr", {"time": 30, "lat": 10, "lon": 10})
    req = {"variables": ["sst"], "bbox": (30, 46, -6, 37),
           "time": ("2000-03-01", "2000-03-31")}
    res = subset_dataset(src, req, tmp_path / "out.zarr")
    assert res["ok"], res
    out = xr.open_zarr(tmp_path / "out.zarr").load()
    assert out.sizes == {"time": 31, "lat": 17, "lon": 44}
    assert float(out.lat.min()) == 30 and float(out.lat.max()) == 46
    assert float(out.lon.min()) == -6 and float(out.lon.max()) == 37
    assert str(out.time.values[0])[:10] == "2000-03-01"
    full = xr.open_zarr(src).load()
    exp = full.sst.sel(time=out.time, lat=out.lat, lon=out.lon)
    np.testing.assert_array_equal(out.sst.values, exp.values)
    assert "geometa_subset_request" in out.attrs


def test_descending_latitude(tmp_path):
    ds = make_ds(np.arange(50, 24, -1), np.arange(-10, 46), DAILY[:10])
    res = subset_dataset(ds, {"bbox": (30, 40, -5, 5)}, tmp_path / "o.zarr")
    assert res["ok"], res
    out = xr.open_zarr(tmp_path / "o.zarr").load()
    assert out.sizes["lat"] == 11
    assert out.lat.values[0] == 40 and out.lat.values[-1] == 30  # order preserved
    np.testing.assert_array_equal(
        out.sst.values, ds.sst.sel(lat=out.lat, lon=out.lon).values)


def test_lon360_dataset_with_request_across_prime_meridian(tmp_path):
    ds = make_ds(np.arange(-90, 91, 2), np.arange(0, 360, 2), DAILY[:5])
    res = subset_dataset(ds, {"bbox": (30, 46, -6, 36)}, tmp_path / "o.zarr")
    assert res["ok"], res
    out = xr.open_zarr(tmp_path / "o.zarr").load()
    # output is in the request's convention, west -> east, monotonic
    assert out.lon.values[0] == -6 and out.lon.values[-1] == 36
    assert np.all(np.diff(out.lon.values) == 2)
    # -6 must come from the dataset's 354
    np.testing.assert_array_equal(
        out.sst.sel(lon=-6).values, ds.sst.sel(lon=354, lat=out.lat).values)
    assert any("convention" in w for w in res["warnings"])


def test_antimeridian_crossing(tmp_path):
    ds = make_ds(np.arange(-20, 21, 2), np.arange(-180, 180, 2), DAILY[:3])
    res = subset_dataset(ds, {"bbox": (-10, 10, 170, -170)}, tmp_path / "o.zarr")
    assert res["ok"], res
    out = xr.open_zarr(tmp_path / "o.zarr").load()
    assert out.lon.values[0] == 170 and out.lon.values[-1] == 190  # continuous
    assert out.sizes["lon"] == 11
    np.testing.assert_array_equal(  # 190 == -170 in the dataset
        out.sst.sel(lon=190).values, ds.sst.sel(lon=-170, lat=out.lat).values)


def test_noleap_calendar(tmp_path):
    time = xr.date_range("2000-01-01", periods=730, freq="D", calendar="noleap",
                         use_cftime=True)
    ds = make_ds(np.arange(0, 10), np.arange(0, 10), time)
    src = tmp_path / "noleap.nc"
    ds.to_netcdf(src)
    res = subset_dataset(str(src), {"time": ("2000-02-27", "2000-03-02")},
                         tmp_path / "o.nc")
    assert res["ok"], res
    out = xr.open_dataset(tmp_path / "o.nc")
    assert out.sizes["time"] == 4  # Feb 27, Feb 28, Mar 1, Mar 2 (no Feb 29 in noleap)
    assert out.time.dt.calendar == "noleap"


def test_date_only_end_is_inclusive_for_hourly_data(tmp_path):
    time = pd.date_range("2020-12-30", "2021-01-02", freq="h")
    ds = make_ds([0, 1], [0, 1], time)
    est = estimate_subset_size(ds, {"time": ("2020-12-31", "2020-12-31")})
    assert est["ok"] and est["shape"]["time"] == 24


def test_open_ended_time(tmp_path):
    ds = make_ds([0, 1], [0, 1], DAILY)
    est = estimate_subset_size(ds, {"time": ("2001-06-01", None)})
    assert est["ok"] and est["shape"]["time"] == len(DAILY[DAILY >= "2001-06-01"])


def test_stride(tmp_path):
    ds = make_ds(np.arange(0, 20), np.arange(0, 40), DAILY[:10])
    est = estimate_subset_size(ds, {"stride": {"lat": 2, "lon": 4}})
    assert est["shape"] == {"time": 10, "lat": 10, "lon": 10}


def test_netcdf_output_roundtrip(tmp_path):
    ds = make_ds(np.arange(0, 10), np.arange(0, 10), DAILY[:10])
    res = subset_dataset(ds, {"bbox": (2, 5, 2, 5)}, tmp_path / "o.nc")
    assert res["ok"] and res["format"] == "netcdf"
    out = xr.open_dataset(tmp_path / "o.nc").load()
    np.testing.assert_array_equal(
        out.sst.values, ds.sst.sel(lat=slice(2, 5), lon=slice(2, 5)).values)


# ------------------------------------------------------------- estimation
def test_estimate_matches_output_and_is_chunk_aware(tmp_path):
    src = to_zarr(make_ds(np.arange(0, 50), np.arange(0, 50), DAILY[:200]),
                  tmp_path / "a.zarr", {"time": 100, "lat": 25, "lon": 25})
    req = {"bbox": (0, 5, 0, 5), "time": ("2000-01-01", "2000-01-10")}
    est = estimate_subset_size(src, req)
    assert est["ok"]
    assert est["output_bytes"] == 10 * 6 * 6 * 8
    # touches 1 time chunk x 1 lat chunk x 1 lon chunk = 100*25*25 float64
    assert est["bytes_read"] == 100 * 25 * 25 * 8
    assert est["read_amplification"] > 1
    res = subset_dataset(src, req, tmp_path / "o.zarr")
    assert res["estimated_output_bytes"] == est["output_bytes"]


def test_estimate_reads_no_data():
    ds = make_ds([0, 1], [0, 1], DAILY[:3])
    est = estimate_subset_size(ds, {"variables": ["sst"]}, max_bytes=10)
    assert est["ok"] and est["within_limit"] is False


# ----------------------------------------------------------------- errors
@pytest.mark.parametrize("req, code, axis", [
    ({"time": ("2030-01-01", "2030-12-31")}, "empty_selection", "time"),
    ({"bbox": (60, 70, 0, 5)}, "empty_selection", "lat"),
    ({"bbox": (0, 1, 100, 120)}, "empty_selection", "lon"),
    ({"variables": ["nope"]}, "unknown_variable", None),
    ({"bbox": (10, 5, 0, 1)}, "bad_bbox", None),
    ({"stride": {"lat": 0}}, "bad_stride", None),
])
def test_structured_errors(req, code, axis):
    ds = make_ds(np.arange(0, 10), np.arange(0, 10), DAILY[:10])
    res = estimate_subset_size(ds, req)
    assert res["ok"] is False and res["error"]["code"] == code
    if axis:
        assert res["error"]["details"]["axis"] == axis
        assert "available" in res["error"]["details"]


def test_too_large_refused_and_nothing_written(tmp_path):
    ds = make_ds(np.arange(0, 10), np.arange(0, 10), DAILY[:10])
    res = subset_dataset(ds, {}, tmp_path / "o.zarr", max_bytes=100)
    assert res["error"]["code"] == "too_large"
    assert not (tmp_path / "o.zarr").exists()


def test_output_exists_guard(tmp_path):
    ds = make_ds([0, 1], [0, 1], DAILY[:2])
    assert subset_dataset(ds, {}, tmp_path / "o.zarr")["ok"]
    assert subset_dataset(ds, {}, tmp_path / "o.zarr")["error"]["code"] == "output_exists"
    assert subset_dataset(ds, {}, tmp_path / "o.zarr", overwrite=True)["ok"]


def test_curvilinear_grid_is_rejected():
    ds = xr.Dataset(
        {"sst": (("y", "x"), np.zeros((3, 3)))},
        coords={"lat": (("y", "x"), np.zeros((3, 3)), {"units": "degrees_north"}),
                "lon": (("y", "x"), np.zeros((3, 3)), {"units": "degrees_east"})})
    res = estimate_subset_size(ds, {"bbox": (0, 1, 0, 1)})
    assert res["error"]["code"] == "unsupported_grid"


# ----------------------------------------------------------- verification
def test_verify_ok_and_request_checks(tmp_path):
    ds = make_ds(np.arange(0, 10), np.arange(0, 10), DAILY[:10])
    req = {"bbox": (2, 5, 2, 5), "time": ("2000-01-02", "2000-01-05")}
    subset_dataset(ds, req, tmp_path / "o.zarr")
    v = verify_subset(str(tmp_path / "o.zarr"), req, expected_units={"sst": "K"})
    assert v["ok"], v
    assert v["summary"]["sst"]["nan_fraction"] == 0
    # same output checked against a *narrower* request must fail
    bad = verify_subset(str(tmp_path / "o.zarr"), {"bbox": (2, 3, 2, 3)})
    assert not bad["ok"]
    assert bad["issues"][0]["code"] == "coords_outside_request"


def test_verify_detects_all_nan_units_and_range(tmp_path):
    ds = make_ds([0, 1], [0, 1], DAILY[:3])
    nan_ds = ds.copy(deep=True)
    nan_ds["sst"][:] = np.nan
    codes = {i["code"] for i in verify_subset(nan_ds)["issues"]}
    assert "all_nan" in codes

    v = verify_subset(ds, expected_units={"sst": "degC"})
    assert "units_mismatch" in {i["code"] for i in v["issues"]}

    v = verify_subset(ds, reference_stats={"sst": {"min": 0.0, "max": 100.0}})
    assert "outside_reference_range" in {i["code"] for i in v["issues"]}  # max ~2e6

    leaked = ds.copy(deep=True)
    leaked["sst"][0, 0, 0] = 9.96921e36  # unmasked netCDF default fill value
    v = verify_subset(leaked, reference_stats={"sst": {"min": 0.0, "max": 3e6}})
    assert not v["ok"]


def test_verify_flags_empty_dimension():
    ds = make_ds([0, 1], [0, 1], DAILY[:3]).isel(lat=slice(0, 0))
    assert not verify_subset(ds)["ok"]


# -------------------------------------------------------- coverage warnings
def warns(ds, req):
    est = estimate_subset_size(ds, req)
    assert est["ok"], est
    return [w for w in est["warnings"] if "partly covered" in w]


def test_period_running_past_the_end_of_the_data_warns():
    """Real run: HadISST ends 2026-06-16, the request was the whole of 2026."""
    ds = make_ds([0, 1], [0, 1], DAILY)                     # 2000-01-01 .. 2001-12-31
    w = warns(ds, {"time": ("2001-06-01", "2002-06-30")})
    assert len(w) == 1 and "ends 2001-12-31" in w[0] and "2002-06-30" in w[0]


def test_period_starting_before_the_data_warns():
    ds = make_ds([0, 1], [0, 1], DAILY)
    w = warns(ds, {"time": ("1999-06-01", "2000-06-30")})
    assert len(w) == 1 and "starts 2000-01-01" in w[0]


def test_period_inside_the_data_does_not_warn():
    ds = make_ds([0, 1], [0, 1], DAILY)
    assert warns(ds, {"time": ("2000-03-01", "2001-03-31")}) == []
    assert warns(ds, {"time": ("2000-01-01", "2001-12-31")}) == []        # exactly the data


def test_monthly_data_stamped_mid_month_does_not_warn_for_the_last_month():
    time = pd.date_range("2000-01-01", periods=12, freq="MS") + pd.Timedelta(days=15)   # 16th
    ds = make_ds([0, 1], [0, 1], time)
    assert warns(ds, {"time": ("2000-01-01", "2000-12-31")}) == []        # gap < one step
    assert len(warns(ds, {"time": ("2000-01-01", "2001-06-30")})) == 1


def test_latitudes_beyond_the_dataset_warn_but_one_cell_of_slack_does_not():
    ds = make_ds(np.arange(30, 46), np.arange(0, 10), DAILY[:3])
    w = warns(ds, {"bbox": (20, 40, 0, 9)})
    assert len(w) == 1 and "latitudes" in w[0] and "[30.0, 45.0]" in w[0]
    assert warns(ds, {"bbox": (29.5, 45.5, 0, 9)}) == []


def test_longitudes_only_partly_covered_warn():
    ds = make_ds(np.arange(0, 5), np.arange(-6, 37), DAILY[:3])
    w = warns(ds, {"bbox": (0, 4, -20, 10)})
    assert len(w) == 1 and "longitudes" in w[0]
    assert warns(ds, {"bbox": (0, 4, -6, 36)}) == []                      # exactly covered
    assert warns(ds, {"bbox": (0, 4, 0, 10)}) == []


def test_global_dataset_never_warns_for_a_regional_box():
    ds = make_ds(np.arange(-90, 91, 2), np.arange(-180, 180, 2), DAILY[:3])
    assert warns(ds, {"bbox": (30, 46, -6, 37)}) == []
    assert warns(ds, {"bbox": (-10, 10, 170, -170)}) == []                 # antimeridian


def test_coverage_warning_reaches_the_subset_result(tmp_path):
    ds = make_ds([0, 1], [0, 1], DAILY)
    res = subset_dataset(ds, {"time": ("2001-06-01", "2002-06-30")}, tmp_path / "o.zarr")
    assert res["ok"] and any("partly covered" in w for w in res["warnings"])


# ------------------------------------------------------------------- extent
def test_result_reports_the_extent_actually_delivered(tmp_path):
    ds = make_ds(np.arange(30, 46), np.arange(-6, 37), DAILY)
    res = subset_dataset(ds, {"bbox": (35, 40, 0, 10), "time": ("2000-03-01", "2000-03-31")},
                         tmp_path / "o.zarr")
    assert res["extent"] == {"time": ["2000-03-01T00:00:00", "2000-03-31T00:00:00"],
                             "lat": [35.0, 40.0], "lon": [0.0, 10.0]}


def test_extent_longitudes_follow_the_request_convention(tmp_path):
    ds = make_ds(np.arange(-90, 91, 2), np.arange(0, 360, 2), DAILY[:3])      # 0-360 dataset
    res = subset_dataset(ds, {"bbox": (30, 46, -6, 36)}, tmp_path / "a.zarr")
    assert res["extent"]["lon"] == [-6.0, 36.0]
    ds = make_ds(np.arange(-20, 21, 2), np.arange(-180, 180, 2), DAILY[:3])
    res = subset_dataset(ds, {"bbox": (-10, 10, 170, -170)}, tmp_path / "b.zarr")
    assert res["extent"]["lon"] == [170.0, 190.0]                              # continuous


def test_extent_exposes_clipping_to_the_available_period(tmp_path):
    """A request for a whole year over data that ends in June delivers Jan-Jun, visibly."""
    ds = make_ds([0, 1], [0, 1], DAILY)                                        # ends 2001-12-31
    res = subset_dataset(ds, {"time": ("2001-01-01", "2001-12-31")}, tmp_path / "o.zarr")
    assert res["extent"]["time"][1].startswith("2001-12-31")
    res = subset_dataset(ds, {"time": ("2001-06-01", "2002-06-30")}, tmp_path / "p.zarr")
    assert res["extent"]["time"] == ["2001-06-01T00:00:00", "2001-12-31T00:00:00"]
