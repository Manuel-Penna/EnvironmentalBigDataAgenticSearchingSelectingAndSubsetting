"""Run with:  pytest -q test_resilience.py

A real run (HadISST on coastwatch.pfeg.noaa.gov) failed with HTTP 429/503 while opening the
source. These tests cover what the code must do in that case: retry once, report
"source_unavailable" (not "open_failed"), and not push the agent to repeat the call."""
import numpy as np
import pytest
from langchain_core.messages import AIMessage

import subsetting as sub
from test_agent import SUBSET_ARGS, call, make_agent, world  # noqa: F401  (fixture)
from test_subsetting import DAILY, make_ds

URL = "http://example.org/erddap/griddap/med"


class FlakyExtractor:
    """Fails with `message` on every open except the first `ok_opens`."""
    message = "HTTP Error occurred 429 Client Error: Too Many Requests for url"
    ok_opens = 0
    opens = 0

    def _open_dataset(self, source):
        type(self).opens += 1
        if type(self).opens <= type(self).ok_opens:
            return make_ds(np.arange(30, 46), np.arange(-6, 37), DAILY)
        raise RuntimeError(type(self).message)


@pytest.fixture
def flaky(monkeypatch):
    monkeypatch.setenv("OPEN_RETRY_DELAY", "0")
    monkeypatch.setenv("OPEN_RETRIES", "1")

    class F(FlakyExtractor):
        pass
    F.opens = 0
    monkeypatch.setattr(sub, "NetCDFExtractor", F)
    return F


def test_transient_error_is_retried_once_then_succeeds(flaky):
    flaky.ok_opens = 99
    flaky.opens = 0
    orig = flaky._open_dataset
    state = {"n": 0}

    def once_then_ok(self, source):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("503 Service Unavailable")
        return orig(self, source)
    flaky._open_dataset = once_then_ok
    ds = sub.open_source(URL, "netcdf")
    assert state["n"] == 2 and "sst" in ds.data_vars


def test_persistent_transient_error_becomes_source_unavailable(flaky):
    with pytest.raises(sub.SubsetError) as e:
        sub.open_source(URL, "netcdf")
    assert e.value.code == "source_unavailable" and e.value.details["attempts"] == 2
    assert flaky.opens == 2 and "later" in e.value.message


def test_retries_can_be_disabled(flaky, monkeypatch):
    monkeypatch.setenv("OPEN_RETRIES", "0")
    with pytest.raises(sub.SubsetError) as e:
        sub.open_source(URL, "netcdf")
    assert e.value.code == "source_unavailable" and flaky.opens == 1


def test_permanent_errors_are_not_retried_or_relabelled(flaky):
    flaky.message = "NetCDF: file not found (404)"
    with pytest.raises(sub.SubsetError) as e:
        sub.open_source(URL, "netcdf")
    assert e.value.code == "open_failed" and flaky.opens == 1


def test_local_sources_are_never_retried(flaky):
    flaky.message = "503 Service Unavailable"
    with pytest.raises(sub.SubsetError) as e:
        sub.open_source("/data/x.nc", "netcdf")
    assert e.value.code == "open_failed" and flaky.opens == 1


def test_estimate_reports_source_unavailable_as_a_tool_error(flaky):
    res = sub.estimate_subset_size(URL, {"variables": ["sst"]}, source_format="netcdf")
    assert res["ok"] is False and res["error"]["code"] == "source_unavailable"


def test_failed_extraction_is_not_nudged_and_is_not_counted_as_agent_failure(world, flaky):
    reg, ctx = world
    rec = ctx.catalog.get("med")
    rec["source"], rec["format"] = URL, "netcdf"
    ctx.share_remote_handles = False         # one open per tool call, as for local-style access
    flaky.ok_opens = 1                       # the size estimate works, the extraction does not
    script = [call("estimate_subset_size", **SUBSET_ARGS),
              call("subset_dataset", **SUBSET_ARGS),
              AIMessage(content="The server is rate-limiting; try again later.")]
    res = make_agent(script, reg)[0].run("t")
    assert res.steps == 3 and res.format_errors == 0           # no extract_nudge round trips
    assert res.tool_calls[1]["error_code"] == "source_unavailable"
    assert res.outcome == "source_unavailable" and res.stop_reason == "final_answer"


def test_never_attempted_extraction_is_still_nudged(world):
    reg, _ = world
    script = [call("estimate_subset_size", **SUBSET_ARGS),
              AIMessage(content="Here is the size."),
              AIMessage(content="Still just the size."),
              AIMessage(content="I decided not to extract.")]
    res = make_agent(script, reg)[0].run("t")
    assert res.steps == 4 and res.answer == "I decided not to extract."   # nudged twice, then accepted
