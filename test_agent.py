"""Run with:  pytest -q test_agent.py

The loop is tested with a scripted fake chat model, so no network or API key is
needed. This checks the LOOP (control flow, guards, logging), not how well a real
model chooses tools: that is what the evaluation set is for.
"""
import json

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import (DEFAULT_GROQ_MODEL, GeoMetaAgent, make_groq_llm,
                   messages_to_dicts)
from agent_tools import ToolContext, build_registry
from catalog_adapters import InMemoryCatalog
from test_agent_tools import record_from_ds
from test_subsetting import DAILY, make_ds


# ------------------------------------------------------------------ fakes
def call(name, **args):
    """An AIMessage that requests one tool call."""
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"id_{name}", "type": "tool_call"}])


class ScriptedChat:
    """Plays back a script. Items: AIMessage | Exception | callable(messages) -> AIMessage."""

    def __init__(self, script):
        self.script = list(script)
        self.seen: list[list] = []          # messages received on each invoke
        self.tools = None
        self.tool_choices: list = []

    def bind_tools(self, tools, tool_choice=None, **kw):
        self.tools = tools
        self.tool_choices.append(tool_choice)
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(messages) if callable(item) else item


def last_json(messages):
    return json.loads(messages[-1].content)


@pytest.fixture
def world(tmp_path):
    ds = make_ds(np.arange(30, 46), np.arange(-6, 37), DAILY)
    ds.to_zarr(tmp_path / "med.zarr")
    rec = record_from_ds(ds, tmp_path / "med.zarr", "med", "Mediterranean sea surface temperature",
                         var_meta={"long_name": "sea surface temperature"})
    ctx = ToolContext(catalog=InMemoryCatalog([rec]), output_dir=tmp_path / "subsets",
                      default_max_bytes=10_000_000)
    return build_registry(ctx), ctx


def make_agent(script, registry, **kw):
    llm = ScriptedChat(script)
    return GeoMetaAgent(llm, registry, **kw), llm


SUBSET_ARGS = {"dataset_id": "med", "variables": ["sst"],
               "bbox": {"lat_min": 35, "lat_max": 40, "lon_min": 0, "lon_max": 10},
               "time": {"start": "2000-03-01", "end": "2000-03-31"}}


# ------------------------------------------------------------------ tests
def test_full_workflow(world):
    reg, ctx = world
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    script = [
        call("search_catalog", query="sea surface temperature"),
        call("get_dataset", dataset_id="med"),
        call("estimate_subset_size", **SUBSET_ARGS),
        call("subset_dataset", **SUBSET_ARGS),
        lambda m: call("verify_subset", subset_id=last_json(m)["subset_id"]),
        AIMessage(content="Extracted the Mediterranean SST subset.", usage_metadata=usage),
    ]
    agent, llm = make_agent(script, reg)
    res = agent.run("Mediterranean SST, March 2000")

    assert res.ok and res.stop_reason == "final_answer" and res.steps == 6
    assert "Mediterranean" in res.answer
    assert [c["name"] for c in res.tool_calls] == [
        "search_catalog", "get_dataset", "estimate_subset_size", "subset_dataset", "verify_subset"]
    assert all(c["ok"] for c in res.tool_calls)
    assert res.subsets == [{"subset_id": res.subsets[0]["subset_id"], "verified": True}]
    assert res.usage == {"input_tokens": 10, "output_tokens": 5}
    # conversation shape: system, user, then assistant/tool pairs
    assert isinstance(res.messages[0], SystemMessage) and isinstance(res.messages[1], HumanMessage)
    assert sum(isinstance(m, ToolMessage) for m in res.messages) == 5
    # provenance log of the registry saw every call too
    assert [e["tool"] for e in ctx.log][-1] == "verify_subset"
    json.dumps(res.to_dict())  # serialisable for the evaluation harness


def test_system_prompt_and_tools_are_passed_to_the_model(world):
    reg, ctx = world
    import datetime
    ctx.today = datetime.date(2026, 10, 19)
    agent, llm = make_agent([AIMessage(content="Which variable do you need?")], reg)
    res = agent.run("hello")
    sys_msg = llm.seen[0][0]
    assert "2026-10-19" in sys_msg.content and "{today}" not in sys_msg.content
    assert {t["function"]["name"] for t in llm.tools} == set(reg.names)
    assert res.ok and res.steps == 1 and res.tool_calls == []   # clarifying question allowed


def test_unverified_subset_is_challenged_before_finishing(world):
    reg, _ = world
    script = [
        call("subset_dataset", **SUBSET_ARGS),
        AIMessage(content="Done!"),                                   # tries to finish early
        lambda m: call("verify_subset", subset_id=json.loads(
            [x for x in m if isinstance(x, ToolMessage)][-1].content)["subset_id"]),
        AIMessage(content="Done, verified."),
    ]
    agent, llm = make_agent(script, reg)
    res = agent.run("task")
    nudge = [m for m in res.messages if isinstance(m, HumanMessage)][-1].content
    assert "verify_subset" in nudge
    assert res.ok and res.answer == "Done, verified." and res.subsets[0]["verified"] is True


def test_verify_nudge_is_limited(world):
    reg, _ = world
    script = [call("subset_dataset", **SUBSET_ARGS)] + [AIMessage(content="Done!")] * 3
    res, _ = (lambda a: (a[0].run("t"), a[1]))(make_agent(script, reg))
    assert res.ok and res.answer == "Done!" and res.subsets[0]["verified"] is None  # not stuck


def test_failed_verification_is_reported_not_hidden(world):
    reg, ctx = world
    script = [
        call("subset_dataset", dataset_id="med", variables=["sst"]),
        lambda m: call("verify_subset", subset_id=last_json(m)["subset_id"]),
        AIMessage(content="Could not verify."),
    ]
    ctx.catalog.get("med")["variables"][0]["statistics"]["max"] = 5.0   # makes verification fail
    res = make_agent(script, reg)[0].run("t")
    assert res.subsets[0]["verified"] is False


def test_tool_errors_are_fed_back_to_the_model(world):
    reg, _ = world
    script = [call("get_dataset", dataset_id="nope"), AIMessage(content="Not found.")]
    agent, llm = make_agent(script, reg)
    res = agent.run("t")
    tool_msg = [m for m in res.messages if isinstance(m, ToolMessage)][0]
    assert json.loads(tool_msg.content)["error"]["code"] == "unknown_dataset"
    assert res.tool_calls[0]["ok"] is False and res.tool_calls[0]["error_code"] == "unknown_dataset"


def test_invalid_arguments_are_fed_back(world):
    reg, _ = world
    script = [call("estimate_subset_size", dataset_id="med"), AIMessage(content="ok")]
    res = make_agent(script, reg)[0].run("t")
    assert res.tool_calls[0]["error_code"] == "invalid_arguments"


def test_repeated_identical_call_gets_a_note(world):
    reg, _ = world
    script = [call("search_catalog", query="sst"), call("search_catalog", query="sst"),
              AIMessage(content="ok")]
    res = make_agent(script, reg)[0].run("t")
    tool_msgs = [json.loads(m.content) for m in res.messages if isinstance(m, ToolMessage)]
    assert "note" not in tool_msgs[0] and "already made" in tool_msgs[1]["note"]


def test_multiple_tool_calls_in_one_message(world):
    reg, _ = world
    two = AIMessage(content="", tool_calls=[
        {"name": "get_dataset", "args": {"dataset_id": "med"}, "id": "a", "type": "tool_call"},
        {"name": "search_catalog", "args": {}, "id": "b", "type": "tool_call"}])
    res = make_agent([two, AIMessage(content="ok")], reg)[0].run("t")
    ids = [m.tool_call_id for m in res.messages if isinstance(m, ToolMessage)]
    assert ids == ["a", "b"] and res.steps == 2


def test_step_limit_forces_a_summary_without_tools(world):
    reg, _ = world
    script = [call("search_catalog", query=f"q{i}") for i in range(3)] + [
        AIMessage(content="I searched three times and found nothing final.")]
    agent, llm = make_agent(script, reg, max_steps=3)
    res = agent.run("t")
    assert res.stop_reason == "max_steps" and not res.ok and res.steps == 3
    assert "searched three times" in res.answer
    assert llm.tool_choices[-1] == "none"            # wrap-up call cannot use tools


def test_malformed_tool_call_error_is_recovered(world):
    reg, _ = world
    bad = Exception("Error code: 400 - {'error': {'code': 'tool_use_failed', "
                    "'message': 'Failed to call a function.'}}")
    script = [bad, call("search_catalog", query="sst"), AIMessage(content="ok")]
    res = make_agent(script, reg)[0].run("t")
    assert res.ok and res.steps == 3
    assert any("malformed" in m.content for m in res.messages if isinstance(m, HumanMessage))


def test_invalid_tool_calls_field_is_recovered(world):
    reg, _ = world
    broken = AIMessage(content="", invalid_tool_calls=[
        {"name": "search_catalog", "args": "{bad json", "id": "1", "error": "bad", "type": "invalid_tool_call"}])
    res = make_agent([broken, AIMessage(content="ok")], reg)[0].run("t")
    assert res.ok and res.steps == 2


def test_malformed_recovery_is_limited(world):
    reg, _ = world
    bad = Exception("tool_use_failed")
    res = make_agent([bad, bad, bad], reg)[0].run("t")
    assert res.stop_reason == "llm_error" and "tool_use_failed" in res.error


def test_other_llm_errors_stop_cleanly(world):
    reg, _ = world
    res = make_agent([RuntimeError("rate limit exceeded")], reg)[0].run("t")
    assert res.stop_reason == "llm_error" and res.answer == "" and "rate limit" in res.error


def test_on_event_receives_steps(world):
    reg, _ = world
    events = []
    script = [call("search_catalog", query="sst"), AIMessage(content="ok")]
    make_agent(script, reg, on_event=events.append)[0].run("t")
    assert [e["type"] for e in events] == ["tool_call"] and events[0]["name"] == "search_catalog"


def test_messages_to_dicts_roundtrips_to_json(world):
    reg, _ = world
    res = make_agent([call("search_catalog"), AIMessage(content="ok")], reg)[0].run("t")
    d = messages_to_dicts(res.messages)
    assert d[0]["role"] == "system" and d[2]["tool_calls"][0]["name"] == "search_catalog"
    json.dumps(d)


# ------------------------------------------------------------ Groq factory
def test_make_groq_llm_needs_a_key(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore the real .env
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        make_groq_llm()


def test_make_groq_llm_model_comes_from_env(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_REASONING_EFFORT", raising=False)
    assert make_groq_llm().model_name == DEFAULT_GROQ_MODEL
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    assert make_groq_llm().model_name == "qwen/qwen3.8-27b"
