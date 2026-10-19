"""
Run the GeoMeta agent from the command line.

    python run_agent.py "Monthly sea surface temperature for the Mediterranean, 2010-2020"
    python run_agent.py "..." --max-mb 200 --max-steps 12 --model openai/gpt-oss-120b

Needs: Qdrant running with an ingested catalog, Ollama serving nomic-embed-text,
and GROQ_API_KEY in a .env file (see .env.example).

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

from agent import GeoMetaAgent, make_groq_llm
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
    ap.add_argument("--model", default=None, help="override GROQ_MODEL")
    ap.add_argument("--quiet", action="store_true", help="do not print the step trace")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(catalog=build_catalog(), output_dir=out,
                      default_max_bytes=int(args.max_mb * 1_000_000),
                      provenance_path=out / "provenance.jsonl")
    llm = make_groq_llm(model=args.model)
    agent = GeoMetaAgent(llm, build_registry(ctx), max_steps=args.max_steps,
                         on_event=None if args.quiet else print_event)

    result = agent.run(args.task)

    print("\n" + (result.answer or "(no answer)"))
    print(f"\n[{result.stop_reason}] {result.steps} steps, {len(result.tool_calls)} tool calls, "
          f"{result.elapsed_s:.1f}s, tokens in/out "
          f"{result.usage['input_tokens']}/{result.usage['output_tokens']}", file=sys.stderr)
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)

    runs = Path(args.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({
        "task": args.task, "model": getattr(llm, "model_name", None),
        "max_mb": args.max_mb, "max_steps": args.max_steps,
        "result": result.to_dict(), "provenance": ctx.log,
    }, indent=2, default=str), encoding="utf-8")
    print(f"run saved to {path}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
