"""
Run the GeoMeta agent from the command line.

    python run_agent.py "Monthly sea surface temperature for the Mediterranean, 2010-2020"
    python run_agent.py "..." --backend ollama --model qwen3:8b --num-ctx 16384
    python run_agent.py "..." --backend groq --model openai/gpt-oss-120b

Needs: Qdrant running with an ingested catalog and Ollama serving nomic-embed-text.
Backend "ollama" (default) also needs a tool-calling model pulled, e.g.
`ollama pull llama3.1:8b`. Backend "groq" needs GROQ_API_KEY in .env (see .env.example).

Every run is saved to runs/<timestamp>.json (task, model, answer, tool calls,
full message trace, token usage): this is the raw material for the evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agent import GeoMetaAgent, make_llm, model_name
from agent_tools import ToolContext, build_registry


def build_catalog():
    # imported here so `--help` works without Qdrant / Ollama installed
    from catalog_adapters import QdrantCatalog
    from embeddings import OllamaEmbedder
    from qdrant_repository import QdrantRepository

    embedder = OllamaEmbedder(model=os.getenv("EMBED_MODEL", "nomic-embed-text"))
    repo = QdrantRepository(embedder=embedder,
                            host=os.getenv("QDRANT_HOST", "localhost"),
                            port=int(os.getenv("QDRANT_PORT", "6333")))
    return QdrantCatalog(repo, embedder)


def print_event(e: dict) -> None:
    if e["type"] == "tool_call":
        args = json.dumps(e["arguments"], default=str)
        status = "ok" if e["ok"] else f"ERROR {e['error_code']}"
        print(f"  [{e['step']:>2}] {e['name']}({args[:110]}) -> {status}", file=sys.stderr)
    elif e["type"] == "thought":
        print(f"  [{e['step']:>2}] {e['text'][:140]}", file=sys.stderr)
    else:
        print(f"  [{e['step']:>2}] ({e['type']})", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", help="what you need, in natural language")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-mb", type=float, default=500, help="subset size limit (MB)")
    ap.add_argument("--out", default="subsets", help="folder for extracted subsets")
    ap.add_argument("--runs-dir", default="runs", help="folder for run records")
    ap.add_argument("--backend", choices=["ollama", "groq"], default=None,
                    help="LLM backend (default: LLM_BACKEND env var, else ollama)")
    ap.add_argument("--model", default=None, help="override OLLAMA_MODEL / GROQ_MODEL")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Ollama context window in tokens (default 16384)")
    ap.add_argument("--max-result-chars", type=int, default=6000,
                    help="cap on each tool result shown to the model (lower for small contexts)")
    ap.add_argument("--quiet", action="store_true", help="do not print the step trace")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(catalog=build_catalog(), output_dir=out,
                      default_max_bytes=int(args.max_mb * 1_000_000),
                      provenance_path=out / "provenance.jsonl")
    extra = {"num_ctx": args.num_ctx} if args.num_ctx and args.backend != "groq" else {}
    llm = make_llm(backend=args.backend, model=args.model, **extra)
    agent = GeoMetaAgent(llm, build_registry(ctx), max_steps=args.max_steps,
                         max_result_chars=args.max_result_chars,
                         on_event=None if args.quiet else print_event)

    result = agent.run(args.task)

    print("\n" + (result.answer or "(no answer)"))
    if result.subsets:   # what the tools reported, so contradictions in the answer are visible
        print("\nTool-reported facts (not written by the model):")
        for sb in result.subsets:
            size = sb.get("output_bytes")
            print(f"  {sb['subset_id']}  verified={sb['verified']}  shape={sb.get('shape')}  "
                  f"{'%.1f KB' % (size / 1024) if size else '?'}  {sb.get('output_path')}")
    print(f"\n[{result.stop_reason} / {result.outcome}] {result.steps} steps, {len(result.tool_calls)} tool calls, "
          f"{result.elapsed_s:.1f}s, tokens in/out "
          f"{result.usage['input_tokens']}/{result.usage['output_tokens']}, "
          f"format errors {result.format_errors}", file=sys.stderr)
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)

    runs = Path(args.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({
        "task": args.task, "backend": args.backend or os.getenv("LLM_BACKEND", "ollama"),
        "model": model_name(llm),
        "max_mb": args.max_mb, "max_steps": args.max_steps,
        "result": result.to_dict(), "provenance": ctx.log,
    }, indent=2, default=str), encoding="utf-8")
    print(f"run saved to {path}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
