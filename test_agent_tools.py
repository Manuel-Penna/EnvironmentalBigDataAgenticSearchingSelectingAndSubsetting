"""Run with:  pytest -q test_agent_tools.py"""
import json

import numpy as np
import pytest
import xarray as xr

from agent_tools import Tool, ToolContext, build_registry, to_text
from catalog_adapters import InMemoryCatalog
from test_subsetting import DAILY, make_ds


def record_from_ds(ds, source, dataset_id, title, fmt="zarr", var_meta=None):
    variables = []
    for name, v in ds.data_vars.items():
        arr = v.values
        variables.append({
            "name": name, "dtype": str(v.dtype), "dimensions": list(v.dims),
            "shape": list(v.shape),
            "attributes": {"units": v.attrs.get("units"), **(var_meta or {})},
            "chunks": [[10], [10], [10]],
            "statistics": {"min": float(arr.min()), "max": float(arr.max()),
                           "mean": float(arr.mean()), "count": int(arr.size), "nan_count": 0},
        })
    return {
        "dataset_id": dataset_id, "source": str(source), "format": fmt, "is_remote": False,
        "global_attributes": {"title": title, "summary": f"{title} test data"},
        "dimensions": {k: int(v) for k, v in ds.sizes.items()},
        "spatial": {"lat_min": float(ds.lat.min()), "lat_max": float(ds.lat.max()),
                    "lon_min": float(ds.lon.min()), "lon_max": float(ds.lon.max())},
        "temporal": {"start": str(ds.time.values[0])[:19], "end": str(ds.time.values[-1])[:19]},
        "variables": variables, "warnings": [],
    }


@pytest.fixture
def env(tmp_path):
    med = make_ds(np.arange(30, 46), np.arange(-6, 37), DAILY)
    pac = make_ds(np.arange(-10, 11), np.arange(-180, 180, 2), DAILY[:50])
    med.to_zarr(tmp_path / "med.zarr")
    pac.to_zarr(tmp_path / "pac.zarr")
    recs = [
        record_from_ds(med, tmp_path / "med.zarr", "med", "Mediterranean sea surface temperature",
                       var_meta={"long_name": "sea surface temperature"}),
        record_from_ds(pac, tmp_path / "pac.zarr", "pac", "Pacific ocean winds",
                       var_meta={"long_name": "wind speed"}),
    ]
    catalog = InMemoryCatalog(recs)
    ctx = ToolContext(catalog=catalog, output_dir=tmp_path / "subsets",
                      default_max_bytes=10_000_000,
                      provenance_path=tmp_path / "prov.jsonl",
                      require_estimate=False)   # strict mode is tested separately below
    return build_registry(ctx), ctx, catalog, tmp_path


# ------------------------------------------------------------------ schemas
def test_schemas_for_both_backends(env):
    reg, *_ = env
    names = {"resolve_time_range", "search_catalog", "get_dataset",
             "estimate_subset_size", "subset_dataset", "verify_subset"}
    assert set(reg.names) == names
    for t in reg.anthropic_tools():
        assert t["input_schema"]["type"] == "object" and t["description"]
    for t in reg.ollama_tools():
        assert t["type"] == "function" and t["function"]["parameters"]["type"] == "object"
    json.dumps(reg.anthropic_tools())  # must be serialisable


# ------------------------------------------------------------------- search
def test_search_by_topic_and_filters(env):
    reg, *_ = env
    r = reg.call("search_catalog", {"query": "sea surface temperature"})
    assert r["ok"] and r["results"][0]["dataset_id"] == "med"
    assert r["results"][0]["variables"][0].startswith("sst")

    r = reg.call("search_catalog", {"query": "wind", "bbox": {
        "lat_min": -5, "lat_max": 5, "lon_min": 170, "lon_max": -170}})  # antimeridian
    assert [h["dataset_id"] for h in r["results"]] == ["pac"]

    r = reg.call("search_catalog", {"bbox": {
        "lat_min": 30, "lat_max": 40, "lon_min": 0, "lon_max": 10}})
    assert [h["dataset_id"] for h in r["results"]] == ["med"]


def test_search_time_filter_open_ended_and_empty_hint(env):
    reg, *_ = env
    r = reg.call("search_catalog", {"time": {"start": "2001-06-01"}})
    assert [h["dataset_id"] for h in r["results"]] == ["med"]  # pac ends 2000-02-19
    r = reg.call("search_catalog", {"query": "temperature", "format": "netcdf"})
    assert r["n_results"] == 0 and "hint" in r
    assert r["applied_filters"]["format"] == "netcdf"  # filters are echoed


# --------------------------------------------------------------- get_dataset
def test_get_dataset_is_compact(env):
    reg, *_ = env
    r = reg.call("get_dataset", {"dataset_id": "med"})
    assert r["ok"] and r["title"].startswith("Mediterranean")
    v = r["variables"][0]
    assert v["name"] == "sst" and v["units"] == "K" and "stats" in v
    assert v["chunk_shape"] == [10, 10, 10]
    assert "global_attributes" not in r
    assert "global_attributes" in reg.call(
        "get_dataset", {"dataset_id": "med", "include_attributes": True})
    bad = reg.call("get_dataset", {"dataset_id": "med", "variables": ["nope"]})
    assert "Unknown variables" in bad["notes"][0]


# ------------------------------------------------------------- full pipeline
def test_estimate_subset_verify_pipeline(env):
    reg, ctx, _, tmp = env
    args = {"dataset_id": "med", "variables": ["sst"],
            "bbox": {"lat_min": 35, "lat_max": 40, "lon_min": 0, "lon_max": 10},
            "time": {"start": "2000-03-01", "end": "2000-03-31"}}
    est = reg.call("estimate_subset_size", args)
    assert est["ok"] and est["within_limit"] and est["shape"] == {"time": 31, "lat": 6, "lon": 11}

    sub = reg.call("subset_dataset", args)
    assert sub["ok"] and sub["subset_id"].startswith("sub_")
    out = xr.open_zarr(sub["output_path"])
    assert out.sizes["time"] == 31
    assert (tmp / "subsets" / f"{sub['subset_id']}.provenance.json").exists()

    v = reg.call("verify_subset", {"subset_id": sub["subset_id"]})
    assert v["ok"] and v["verified"], v

    # same request -> same subset_id (reproducible), re-run overwrites cleanly
    assert reg.call("subset_dataset", args)["subset_id"] == sub["subset_id"]

    # provenance: every call logged, JSONL on disk
    tools_called = [e["tool"] for e in ctx.log]
    assert tools_called[:3] == ["estimate_subset_size", "subset_dataset", "verify_subset"]
    lines = (tmp / "prov.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(ctx.log) and json.loads(lines[1])["summary"]["subset_id"]


def test_verify_in_a_fresh_session_reloads_sidecar(env):
    reg, ctx, catalog, tmp = env
    sid = reg.call("subset_dataset", {"dataset_id": "med", "variables": ["sst"],
                                      "stride": {"lat": 2}})["subset_id"]
    ctx2 = ToolContext(catalog=catalog, output_dir=tmp / "subsets")
    assert build_registry(ctx2).call("verify_subset", {"subset_id": sid})["verified"]


def test_verify_catches_values_outside_catalog_range(env):
    reg, ctx, catalog, _ = env
    sid = reg.call("subset_dataset", {"dataset_id": "med", "variables": ["sst"]})["subset_id"]
    catalog.get("med")["variables"][0]["statistics"]["max"] = 5.0  # pretend range is tiny
    v = reg.call("verify_subset", {"subset_id": sid})
    assert v["ok"] and not v["verified"]
    assert v["issues"][0]["code"] == "outside_reference_range"


# ------------------------------------------------------------ size guidance
def test_too_large_gives_hints_and_subset_refuses(env):
    reg, ctx, *_ = env
    args = {"dataset_id": "med", "variables": ["sst"], "max_bytes": 10_000}
    est = reg.call("estimate_subset_size", args)
    assert est["ok"] and not est["within_limit"]
    assert any("stride" in h for h in est["hints"])
    sub = reg.call("subset_dataset", args)
    assert sub["error"]["code"] == "too_large"
    assert "estimate" in sub["error"]["details"]
    assert not list(ctx.output_dir.glob("*.zarr"))


def test_non_overlapping_request_error_has_dataset_range(env):
    reg, *_ = env
    r = reg.call("estimate_subset_size", {
        "dataset_id": "med", "variables": ["sst"],
        "time": {"start": "2020-01-01", "end": "2020-12-31"}})
    assert r["error"]["code"] == "empty_selection"
    assert r["error"]["details"]["available"][0].startswith("2000-01-01")


# ------------------------------------------------- argument validation/coercion
def test_small_model_style_arguments_are_coerced(env):
    reg, ctx, *_ = env
    r = reg.call("estimate_subset_size", {
        "dataset_id": "med",
        "variables": "sst",                                   # string instead of list
        "bbox": json.dumps({"lat_min": "35", "lat_max": "40",  # stringified, numbers as strings
                            "lon_min": "0", "lon_max": "10"}),
        "time": None,                                         # null optional
        "stride": {"lat": "2"},                               # "2" -> 2
    })
    assert r["ok"], r
    assert ctx.log[-1]["arguments"]["variables"] == ["sst"]   # coerced args are what is logged


@pytest.mark.parametrize("args, fragment", [
    ({"dataset_id": "med"}, "missing required property 'variables'"),
    ({"dataset_id": "med", "variables": ["sst"], "region": "x"}, "unexpected property 'region'"),
    ({"dataset_id": "med", "variables": ["sst"],
      "bbox": {"lat_min": 95, "lat_max": 99, "lon_min": 0, "lon_max": 1}}, "<= 90"),
    ({"dataset_id": "med", "variables": ["sst"], "stride": {"lat": 0}}, ">= 1"),
    ({"dataset_id": "med", "variables": ["sst"], "bbox": {"lat_min": 1}}, "missing required"),
])
def test_invalid_arguments_report_all_problems(env, args, fragment):
    reg, *_ = env
    r = reg.call("estimate_subset_size", args)
    assert r["error"]["code"] == "invalid_arguments" and fragment in r["error"]["message"]
    assert "expected_parameters" in r["error"]["details"]


def test_enum_is_case_insensitive_and_limit_bounds(env):
    reg, *_ = env
    assert reg.call("search_catalog", {"format": "ZARR"})["ok"]
    assert reg.call("search_catalog", {"limit": 500})["error"]["code"] == "invalid_arguments"


# ------------------------------------------------------- robustness
def test_unknown_tool_dataset_and_subset(env):
    reg, *_ = env
    assert reg.call("fly_to_moon")["error"]["code"] == "unknown_tool"
    assert reg.call("get_dataset", {"dataset_id": "zzz"})["error"]["code"] == "unknown_dataset"
    assert reg.call("verify_subset", {"subset_id": "sub_x"})["error"]["code"] == "unknown_subset"


def test_arguments_may_arrive_as_json_string(env):
    reg, *_ = env
    assert reg.call("get_dataset", '{"dataset_id": "med"}')["ok"]


def test_crashing_tool_never_raises(env):
    reg, ctx, *_ = env
    reg.register(Tool("boom", "x", {"type": "object", "properties": {}},
                      lambda ctx: 1 / 0))
    r = reg.call("boom")
    assert r["error"]["code"] == "tool_crashed" and "ZeroDivisionError" in r["error"]["message"]
    assert ctx.log[-1]["error_code"] == "tool_crashed"


def test_to_text_caps_length(env):
    reg, *_ = env
    r = reg.call("search_catalog", {})
    assert len(to_text(r, max_chars=100)) < 200
    assert json.loads(to_text(r))["ok"]


# ------------------------------------------------------ resolve_time_range
def test_resolve_time_range(env):
    import datetime
    reg, ctx, *_ = env
    ctx.today = datetime.date(2026, 10, 19)
    assert reg.call("resolve_time_range", {"year": 2019})["start"] == "2019-01-01"
    r = reg.call("resolve_time_range", {"year": "2019"})          # coerced
    assert (r["start"], r["end"]) == ("2019-01-01", "2019-12-31")
    r = reg.call("resolve_time_range", {"last_amount": 5, "last_unit": "years"})
    assert (r["start"], r["end"]) == ("2021-10-19", "2026-10-19")
    r = reg.call("resolve_time_range", {"last_amount": 3, "last_unit": "months"})
    assert r["start"] == "2026-07-19"
    assert reg.call("resolve_time_range", {})["error"]["code"] == "bad_request"
    assert reg.call("resolve_time_range", {"year": 2019, "last_amount": 2,
                                           "last_unit": "years"})["error"]["code"] == "bad_request"
    assert reg.call("resolve_time_range", {"last_unit": "weeks", "last_amount": 1}
                    )["error"]["code"] == "invalid_arguments"


def test_wrapping_arguments_in_a_filters_object_gets_an_actionable_error(env):
    """Seen in a real llama3.1 run: {"query": ..., "filters": {...}}."""
    reg, *_ = env
    r = reg.call("search_catalog", {"query": "sst", "filters": {"region": "Naples"}})
    msg = r["error"]["message"]
    assert r["error"]["code"] == "invalid_arguments"
    assert "unexpected property 'filters'" in msg and "top-level" in msg


# ------------------------------------------- estimate-first rule (strict mode)
EST_ARGS = {"dataset_id": "med", "variables": ["sst"],
            "bbox": {"lat_min": 35, "lat_max": 40, "lon_min": 0, "lon_max": 10},
            "time": {"start": "2000-03-01", "end": "2000-03-31"}}


@pytest.fixture
def strict(env):
    _, _, catalog, tmp = env
    ctx = ToolContext(catalog=catalog, output_dir=tmp / "strict")   # require_estimate=True
    return build_registry(ctx), ctx


def test_subset_without_any_estimate_is_refused(strict):
    reg, ctx = strict
    r = reg.call("subset_dataset", EST_ARGS)
    assert r["error"]["code"] == "not_estimated" and "estimate_subset_size" in r["error"]["message"]
    assert not list(ctx.output_dir.glob("*.zarr"))


def test_dropping_an_estimated_argument_is_refused(strict):
    """Real run: estimate used a time range, the extraction call left it out, so the whole
    1870-2026 history was extracted while the answer reported the estimated shape."""
    reg, ctx = strict
    assert reg.call("estimate_subset_size", EST_ARGS)["ok"]
    dropped = {k: v for k, v in EST_ARGS.items() if k != "time"}
    r = reg.call("subset_dataset", dropped)
    assert r["error"]["code"] == "not_estimated"
    assert any(d.startswith("time:") for d in r["error"]["details"]["differences"])
    assert r["error"]["details"]["last_estimated_request"]["time"] == ("2000-03-01", "2000-03-31")
    assert not list(ctx.output_dir.glob("*.zarr"))                      # nothing was written
    assert reg.call("subset_dataset", EST_ARGS)["ok"]                    # the estimated request works


def test_changed_bbox_and_variables_are_refused_too(strict):
    reg, _ = strict
    reg.call("estimate_subset_size", EST_ARGS)
    wider = {**EST_ARGS, "bbox": {"lat_min": 30, "lat_max": 40, "lon_min": 0, "lon_max": 10}}
    r = reg.call("subset_dataset", wider)
    assert any(d.startswith("bbox:") for d in r["error"]["details"]["differences"])


def test_matching_request_is_accepted_whatever_the_order_or_limits(strict):
    reg, _ = strict
    reg.call("estimate_subset_size", {**EST_ARGS, "variables": ["sst"]})
    again = {**EST_ARGS, "max_bytes": 5_000_000, "output_format": "netcdf"}   # not part of the identity
    assert reg.call("subset_dataset", again)["ok"]


def test_over_limit_estimate_then_same_subset_reports_too_large(strict):
    reg, _ = strict
    args = {**EST_ARGS, "max_bytes": 1000}
    assert reg.call("estimate_subset_size", args)["within_limit"] is False
    assert reg.call("subset_dataset", args)["error"]["code"] == "too_large"   # not not_estimated


def test_rule_can_be_switched_off(env):
    reg, ctx, *_ = env              # fixture runs with require_estimate=False
    assert ctx.require_estimate is False
    assert reg.call("subset_dataset", {**EST_ARGS, "variables": ["sst"]})["ok"]
