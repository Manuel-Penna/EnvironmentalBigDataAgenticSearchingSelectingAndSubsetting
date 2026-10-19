"""
The GeoMeta agent loop (step 5).

    from agent import GeoMetaAgent, make_groq_llm
    agent = GeoMetaAgent(make_groq_llm(), build_registry(ctx))
    result = agent.run("Monthly sea surface temperature for the Mediterranean, "
                       "2010-2020, under 500 MB")
    print(result.answer)

Design
------
* The loop is explicit (about 60 lines) instead of a LangChain agent executor, so
  every step is visible, logged and countable for the evaluation chapter.
* LangChain is used for what it is good at: one interface over chat models.
  `llm` can be any chat model with tool calling (ChatGroq, ChatAnthropic,
  ChatOllama, ...); nothing else in this file changes.
* The LLM decides WHICH tool to call and WITH WHAT arguments. Code enforces the
  rules that must not depend on model behaviour:
    - step cap, with a forced wrap-up answer when it is hit;
    - malformed tool calls are bounced back to the model, not crashes;
    - an identical repeated call gets a note that nothing will change;
    - the agent cannot finish with a subset it never verified.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)

from agent_tools import ToolRegistry, to_text

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are the GeoMeta agent. You help users find environmental datasets in a catalog of NetCDF and Zarr files and extract exactly the subset they need. Today is {today}.

Work in this order:
1. Understand the request: topic or variable, region, time period, and any size limit. Turn place names into an approximate bounding box (degrees; longitudes are east, west edge first; use lon_min > lon_max when the region crosses the antimeridian). For a calendar year or a relative period ("the last 5 years"), call resolve_time_range. Never work out dates yourself.
2. Call search_catalog. Put only the topic in `query`; region, period and format go in the filters. If nothing is found, relax the filters one at a time (for example drop the period, then the region) and tell the user what you relaxed.
3. Call get_dataset on the 2 or 3 most promising candidates and compare them: does the dataset cover the whole requested region and period? Is the variable the right one (name, long_name, units)? How many NaN values, and how fine is the grid? Prefer full coverage over partial coverage.
4. Call estimate_subset_size before extracting. If the result exceeds the limit, narrow the request or add a stride, following the hints, and tell the user what you reduced.
5. Call subset_dataset, then ALWAYS call verify_subset. If verified is false, do not present the subset as a result: read the issues, then try different bounds or another candidate.
6. Give a final answer: the dataset you chose and why; what was extracted (variables, region, period, shape, size, subset_id and output path); caveats (partial coverage, reductions, warnings); alternatives you considered.

Rules:
- Use only dataset_ids, variable names and values that tool results returned. Never invent them.
- Tool errors tell you what to fix. Read error.message and error.details and adapt. Do not repeat a call that already failed with the same arguments.
- If the request lacks something you cannot reasonably assume (for example no variable or topic at all), reply with one short clarifying question instead of calling tools.
- If no suitable dataset exists, say so plainly and explain what you tried. Do not extract a poor match without telling the user.
- Be concise. The final answer is a normal message without a tool call."""

MALFORMED_NUDGE = ("Your last tool call was malformed and was not executed. Call the tool "
                   "again with valid JSON arguments that match its schema.")
REPEAT_NOTE = ("You already made this exact call and the result will not change. Choose a "
               "different action or give your final answer.")
WRAPUP_PROMPT = ("The step limit has been reached. Without calling any tool, summarise what "
                 "you did, what you found, and what is still missing.")


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
def _text(content: Any) -> str:
    """Message content as plain text (handles providers that return blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return "" if content is None else str(content)


def messages_to_dicts(messages: list[BaseMessage]) -> list[dict]:
    out = []
    for m in messages:
        d: dict[str, Any] = {"role": m.type, "content": _text(m.content)}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [{"name": t["name"], "args": t["args"]} for t in m.tool_calls]
        if m.type == "tool":
            d["name"] = getattr(m, "name", None)
        out.append(d)
    return out


@dataclass
class AgentResult:
    answer: str
    stop_reason: str                    # "final_answer" | "max_steps" | "llm_error"
    steps: int                          # loop iterations (LLM calls attempted)
    tool_calls: list[dict]              # {"step","name","arguments","ok","error_code"}
    subsets: list[dict]                 # {"subset_id","verified": True|False|None}
    usage: dict                         # {"input_tokens","output_tokens"} when reported
    elapsed_s: float
    messages: list[BaseMessage] = field(repr=False, default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.stop_reason == "final_answer"

    def to_dict(self) -> dict:
        return {"answer": self.answer, "stop_reason": self.stop_reason, "steps": self.steps,
                "tool_calls": self.tool_calls, "subsets": self.subsets, "usage": self.usage,
                "elapsed_s": round(self.elapsed_s, 2), "error": self.error,
                "trace": messages_to_dicts(self.messages)}


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class GeoMetaAgent:
    def __init__(self, llm: Any, registry: ToolRegistry, *, max_steps: int = 15,
                 system_prompt: str = SYSTEM_PROMPT, max_result_chars: int = 6000,
                 on_event: Callable[[dict], None] | None = None):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.max_result_chars = max_result_chars
        self.on_event = on_event or (lambda e: None)

    def run(self, task: str) -> AgentResult:
        t0 = time.perf_counter()
        today = (self.registry.ctx.today or date.today()).isoformat()
        messages: list[BaseMessage] = [
            SystemMessage(self.system_prompt.replace("{today}", today)),
            HumanMessage(task),
        ]
        tools = self.registry.ollama_tools()  # OpenAI-style dicts: accepted by bind_tools
        llm = self.llm.bind_tools(tools)

        calls: list[dict] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        seen: dict[str, int] = {}
        created: list[str] = []
        verified: dict[str, bool] = {}
        malformed = unverified_nudges = 0
        step = 0

        def finish(reason: str, answer: str = "", error: str | None = None) -> AgentResult:
            return AgentResult(
                answer=answer, stop_reason=reason, steps=step, tool_calls=calls,
                subsets=[{"subset_id": s, "verified": verified.get(s)} for s in created],
                usage=usage, elapsed_s=time.perf_counter() - t0, messages=messages, error=error)

        for step in range(1, self.max_steps + 1):
            # ---- ask the model ------------------------------------------------
            try:
                ai: AIMessage = llm.invoke(messages)
            except Exception as e:  # noqa: BLE001
                if _is_malformed_tool_call(e) and malformed < 2:
                    malformed += 1
                    messages.append(HumanMessage(MALFORMED_NUDGE))
                    self.on_event({"type": "malformed_tool_call", "step": step})
                    continue
                return finish("llm_error", error=f"{type(e).__name__}: {e}")

            um = getattr(ai, "usage_metadata", None) or {}
            usage["input_tokens"] += um.get("input_tokens", 0)
            usage["output_tokens"] += um.get("output_tokens", 0)
            messages.append(ai)
            if _text(ai.content).strip() and ai.tool_calls:
                self.on_event({"type": "thought", "step": step, "text": _text(ai.content)})

            # ---- no tool call: a final answer (if the rules allow it) ----------
            if not ai.tool_calls:
                if getattr(ai, "invalid_tool_calls", None) and malformed < 2:
                    malformed += 1
                    messages.append(HumanMessage(MALFORMED_NUDGE))
                    self.on_event({"type": "malformed_tool_call", "step": step})
                    continue
                pending = [s for s in created if s not in verified]
                if pending and unverified_nudges < 2:
                    unverified_nudges += 1
                    messages.append(HumanMessage(
                        f"You extracted {pending} but did not call verify_subset on it. "
                        "Verify it before giving your answer."))
                    self.on_event({"type": "verify_nudge", "step": step, "subsets": pending})
                    continue
                return finish("final_answer", answer=_text(ai.content))

            # ---- execute the tool calls ------------------------------------
            for i, tc in enumerate(ai.tool_calls):
                name, args = tc["name"], tc.get("args") or {}
                call_id = tc.get("id") or f"call_{step}_{i}"
                key = name + json.dumps(args, sort_keys=True, default=str)
                seen[key] = seen.get(key, 0) + 1

                result = self.registry.call(name, args)
                if seen[key] > 1:
                    result = {**result, "note": REPEAT_NOTE}
                if name == "subset_dataset" and result.get("ok"):
                    created.append(result["subset_id"])
                if name == "verify_subset" and result.get("ok"):
                    verified[result["subset_id"]] = bool(result["verified"])

                code = (result.get("error") or {}).get("code")
                calls.append({"step": step, "name": name, "arguments": args,
                              "ok": bool(result.get("ok")), "error_code": code})
                self.on_event({"type": "tool_call", **calls[-1]})
                messages.append(ToolMessage(content=to_text(result, self.max_result_chars),
                                            tool_call_id=call_id, name=name))

        # ---- step limit: ask for a summary instead of returning nothing -------
        messages.append(HumanMessage(WRAPUP_PROMPT))
        try:
            ai = self.llm.bind_tools(tools, tool_choice="none").invoke(messages)
            messages.append(ai)
            answer = _text(ai.content)
        except Exception as e:  # noqa: BLE001
            return finish("max_steps", error=f"wrap-up failed: {type(e).__name__}: {e}")
        return finish("max_steps", answer=answer)


def _is_malformed_tool_call(e: Exception) -> bool:
    """Groq returns HTTP 400 `tool_use_failed` when the model emits an invalid call."""
    msg = str(e).lower()
    return "tool_use_failed" in msg or "failed to call a function" in msg


# --------------------------------------------------------------------------
# LLM factory
# --------------------------------------------------------------------------
def make_groq_llm(model: str | None = None, temperature: float = 0.0, **kwargs: Any):
    """ChatGroq configured from the environment (.env is loaded automatically).

    GROQ_API_KEY            required
    GROQ_MODEL              model id (default: openai/gpt-oss-120b). Groq retires model
                            ids regularly: check console.groq.com/docs/deprecations
    GROQ_REASONING_EFFORT   optional (low | medium | high); reasoning models only
    """
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Put it in a .env file next to run_agent.py.")
    from langchain_groq import ChatGroq

    if os.getenv("GROQ_REASONING_EFFORT") and "reasoning_effort" not in kwargs:
        kwargs["reasoning_effort"] = os.environ["GROQ_REASONING_EFFORT"]
    return ChatGroq(model=model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                    temperature=temperature, max_retries=3, **kwargs)
