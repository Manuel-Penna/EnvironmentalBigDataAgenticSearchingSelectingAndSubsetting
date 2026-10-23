"""Run with:  pytest -q test_shared_handle.py

A remote source is opened once per agent run and shared by estimate_subset_size and
subset_dataset (public servers rate-limit), then closed when the run ends."""
import numpy as np
import pytest
import xarray as xr
from langchain_core.messages import AIMessage

import subsetting as sub
from test_agent import SUBSET_ARGS, call, last_json, make_agent, world  # noqa: F401  (fixture)
from test_subsetting import DAILY, make_ds

URL = "http://example.org/erddap/griddap/med"


class CountingExtractor:
    opens = 0
    closed = 0
    fail_next = 0

    def _open_dataset(self, source):
        cls = type(self)
        cls.opens += 1
        if cls.fail_next:
            cls.fail_next -= 1
            raise RuntimeError("HTTP Error 429 Too Many Requests")
        ds = make_ds(np.arange(30, 46), np.arange(-6, 37), DAILY)

        def on_close():
            cls.closed += 1
        ds.set_close(on_close)
        return ds


@pytest.fixture
def remote(world, monkeypatch):
    monkeypatch.setenv("OPEN_RETRIES", "0")

    class C(CountingExtractor):
        pass
    C.opens = C.closed = C.fail_next = 0
    monkeypatch.setattr(sub, "NetCDFExtractor", C)
    reg, ctx = world
    rec = ctx.catalog.get("med")
    rec["source"], rec["format"] = URL, "netcdf"
    return reg, ctx, C


def test_estimate_and_subset_open_the_remote_source_once(remote):
    reg, ctx, C = remote
    script = [call("estimate_subset_size", **SUBSET_ARGS),
              call("subset_dataset", **SUBSET_ARGS),
              lambda m: call("verify_subset", subset_id=last_json(m)["subset_id"]),
              AIMessage(content="done")]
    res = make_agent(script, reg)[0].run("t")
    assert res.outcome == "delivered"
    assert C.opens == 1                      # was 2: one open per tool call
    assert C.closed == 1 and ctx._open == {}  # released when the run ended


def test_output_provenance_names_the_real_source_not_in_memory(remote):
    reg, ctx, C = remote
    assert reg.call("estimate_subset_size", SUBSET_ARGS)["ok"]
    out = reg.call("subset_dataset", SUBSET_ARGS)
    assert out["ok"]
    written = xr.open_zarr(out["output_path"])
    assert written.attrs["geometa_subset_source"] == URL
    assert ctx.subsets[out["subset_id"]]["source"] == URL
    ctx.close()


def test_failed_open_is_not_cached_so_a_later_call_can_succeed(remote):
    reg, ctx, C = remote
    C.fail_next = 1
    first = reg.call("estimate_subset_size", SUBSET_ARGS)
    assert first["ok"] is False and first["error"]["code"] == "source_unavailable"
    assert reg.call("estimate_subset_size", SUBSET_ARGS)["ok"]
    assert C.opens == 2
    ctx.close()


def test_handles_are_closed_even_if_the_run_ends_in_an_error(remote):
    reg, ctx, C = remote
    script = [call("estimate_subset_size", **SUBSET_ARGS), RuntimeError("model down")]
    res = make_agent(script, reg)[0].run("t")
    assert res.stop_reason == "llm_error" and C.closed == 1 and ctx._open == {}


def test_sharing_can_be_switched_off(remote):
    reg, ctx, C = remote
    ctx.share_remote_handles = False
    assert reg.call("estimate_subset_size", SUBSET_ARGS)["ok"]
    assert reg.call("estimate_subset_size", SUBSET_ARGS)["ok"]
    assert C.opens == 2 and ctx._open == {}


def test_local_sources_are_not_shared(world):
    reg, ctx = world
    assert reg.call("estimate_subset_size", SUBSET_ARGS)["ok"]
    assert ctx._open == {}


def test_subset_result_is_identical_with_and_without_sharing(remote):
    reg, ctx, C = remote
    assert reg.call("estimate_subset_size", SUBSET_ARGS)["ok"]
    shared = reg.call("subset_dataset", SUBSET_ARGS)
    ctx.close()
    ctx.share_remote_handles = False
    assert reg.call("estimate_subset_size", SUBSET_ARGS)["ok"]
    plain = reg.call("subset_dataset", SUBSET_ARGS)
    assert shared["subset_id"] == plain["subset_id"] and shared["shape"] == plain["shape"]
    a, b = xr.open_zarr(shared["output_path"]).load(), xr.open_zarr(plain["output_path"]).load()
    xr.testing.assert_identical(a.drop_attrs() if hasattr(a, "drop_attrs") else a, b.drop_attrs() if hasattr(b, "drop_attrs") else b)
