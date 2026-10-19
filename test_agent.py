"""Run with:  pytest -q test_agent.py

The loop is tested with a scripted fake chat model, so no network or API key is
needed. This checks the LOOP (control flow, guards, logging), not how well a real
model chooses tools: that is what the evaluation set is for.
"""
import json

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import (DEFAULT_GROQ_MODEL, DEFAULT_OLLAMA_MODEL, SYSTEM_PROMPT, GeoMetaAgent,
                   make_groq_llm, make_llm, make_ollama_llm, messages_to_dicts, model_name)
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
    res = make_agent([broken, call("search_catalog"), AIMessage(content="ok")], reg)[0].run("t")
    assert res.ok and res.steps == 3 and res.format_errors == 1


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


# ------------------------------------------- guards for small local models
def test_tool_call_written_as_text_is_bounced(world):
    reg, _ = world
    as_json = AIMessage(content='{"name": "search_catalog", "parameters": {"query": "sst"}}')
    as_fence = AIMessage(content='```json\n{"name": "get_dataset", "arguments": {"dataset_id": "med"}}\n```')
    as_tag = AIMessage(content='<tool_call>{"name": "search_catalog"}</tool_call>')
    script = [as_json, as_fence, call("search_catalog", query="sst"), AIMessage(content="done")]
    res = make_agent(script, reg)[0].run("t")
    assert res.ok and res.answer == "done"
    assert res.format_errors == 2 and [c["name"] for c in res.tool_calls] == ["search_catalog"]
    assert any("plain text" in m.content for m in res.messages if isinstance(m, HumanMessage))
    # the XML-ish form is recognised too
    res = make_agent([as_tag, call("search_catalog"), AIMessage(content="done")], reg)[0].run("t")
    assert res.format_errors == 1


def test_text_tool_call_nudges_are_limited(world):
    reg, _ = world
    txt = AIMessage(content='{"name": "search_catalog", "parameters": {}}')
    res = make_agent([txt, txt, txt, AIMessage(content="x")], reg)[0].run("t")
    assert res.format_errors == 2 and res.stop_reason == "final_answer"   # eventually accepted


def test_normal_json_answer_is_not_mistaken_for_a_tool_call(world):
    reg, _ = world
    script = [call("search_catalog"), AIMessage(content='{"dataset": "med", "name": "sst"}')]
    assert make_agent(script, reg)[0].run("t").format_errors == 0


def test_answering_from_memory_is_challenged_once(world):
    reg, _ = world
    script = [AIMessage(content="The best dataset is ERA5."),        # hallucinated, no tools
              call("search_catalog", query="sst"),
              AIMessage(content="Found med.")]
    res = make_agent(script, reg)[0].run("t")
    assert res.answer == "Found med." and len(res.tool_calls) == 1
    # if the model insists, the second no-tool answer is accepted (no endless loop)
    res = make_agent([AIMessage(content="ERA5."), AIMessage(content="ERA5.")], reg)[0].run("t")
    assert res.ok and res.answer == "ERA5." and res.tool_calls == []


def test_wrapup_survives_a_backend_that_ignores_tool_choice(world):
    """ChatOllama ignores tool_choice='none': the wrap-up may still come back as a tool call."""
    reg, _ = world
    script = [call("search_catalog", query="q0"), call("search_catalog", query="q1"),
              call("search_catalog", query="still going"),       # wrap-up attempt 1: rejected
              AIMessage(content="Summary of what I did.")]       # attempt 2: accepted
    res = make_agent(script, reg, max_steps=2)[0].run("t")
    assert res.stop_reason == "max_steps" and res.answer == "Summary of what I did."


def test_wrapup_failure_is_reported(world):
    reg, _ = world
    script = [call("search_catalog", query="q0"), RuntimeError("boom"), RuntimeError("boom")]
    res = make_agent(script, reg, max_steps=1)[0].run("t")
    assert res.stop_reason == "max_steps" and res.answer == "" and "boom" in res.error


def test_baseline_prompt_fits_a_small_context_window(world):
    reg, _ = world
    chars = len(SYSTEM_PROMPT) + len(json.dumps(reg.ollama_tools()))
    assert chars / 4 < 3500, "system prompt + tool schemas should stay well under num_ctx"


# ------------------------------------------------------------ Ollama factory
def clean_env(monkeypatch, *names):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore the real .env
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_make_ollama_llm_defaults_and_env(monkeypatch):
    clean_env(monkeypatch, "OLLAMA_MODEL", "OLLAMA_NUM_CTX", "OLLAMA_HOST", "OLLAMA_KEEP_ALIVE")
    llm = make_ollama_llm()                       # no server needed to construct
    assert llm.model == DEFAULT_OLLAMA_MODEL and llm.num_ctx == 16384 and llm.temperature == 0.0
    assert llm.keep_alive == "30m"
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    llm = make_ollama_llm()
    assert (llm.model, llm.num_ctx, llm.base_url) == ("qwen3:8b", 8192, "http://gpu-box:11434")
    assert make_ollama_llm(model="llama3.2:3b", num_ctx=4096).num_ctx == 4096


def test_ollama_model_accepts_the_tool_schemas(monkeypatch):
    clean_env(monkeypatch, "OLLAMA_MODEL", "OLLAMA_HOST")
    from agent_tools import ToolContext, build_registry
    from catalog_adapters import InMemoryCatalog
    import tempfile
    reg = build_registry(ToolContext(InMemoryCatalog([]), tempfile.mkdtemp()))
    bound = make_ollama_llm().bind_tools(reg.ollama_tools())        # must not raise
    assert len(bound.kwargs["tools"]) == len(reg.names)


def test_make_llm_selects_backend(monkeypatch):
    clean_env(monkeypatch, "LLM_BACKEND", "OLLAMA_MODEL", "GROQ_MODEL", "GROQ_REASONING_EFFORT")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert type(make_llm()).__name__ == "ChatOllama"                # local by default
    assert type(make_llm("groq")).__name__ == "ChatGroq"
    monkeypatch.setenv("LLM_BACKEND", "groq")
    assert type(make_llm()).__name__ == "ChatGroq"
    assert model_name(make_llm("ollama", model="qwen3:8b")) == "qwen3:8b"
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        make_llm("nope")


def test_ollama_wire_format_against_a_fake_server(world):
    """The real ChatOllama client talking to a minimal fake Ollama HTTP server."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    reg, _ = world
    requests = []
    replies = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search_catalog",
                          "arguments": {"query": "sea surface temperature"}}}]}},
        {"message": {"role": "assistant", "content": "Found dataset med. Anything else?"}},
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            out = {"model": body.get("model"), "created_at": "2026-01-01T00:00:00Z",
                   "done": True, "done_reason": "stop", "prompt_eval_count": 100,
                   "eval_count": 20, **replies[len(requests) - 1]}
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write((json.dumps(out) + "\n").encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        llm = make_ollama_llm(model="llama3.1:8b", num_ctx=12000,
                              base_url=f"http://127.0.0.1:{server.server_port}")
        res = GeoMetaAgent(llm, reg).run("sea surface temperature please")
    finally:
        server.shutdown()

    assert res.ok and res.steps == 2 and [c["name"] for c in res.tool_calls] == ["search_catalog"]
    assert res.usage == {"input_tokens": 200, "output_tokens": 40}
    first, second = requests
    assert first["options"]["num_ctx"] == 12000          # the context window reaches the server
    assert len(first["tools"]) == len(reg.names)
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert json.loads(second["messages"][-1]["content"])["n_results"] == 1


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
