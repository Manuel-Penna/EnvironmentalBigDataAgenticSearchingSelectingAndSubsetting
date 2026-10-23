"""
Agent tool layer for GeoMeta.

    ctx = ToolContext(catalog=..., output_dir="subsets")
    registry = build_registry(ctx)

    registry.anthropic_tools()   # list for the Anthropic API  (input_schema)
    registry.ollama_tools()      # list for ollama.chat(tools=...)
    registry.call("search_catalog", {"query": "sea surface temperature"})  # -> dict

Rules every tool follows
------------------------
* Takes JSON-schema-described arguments, returns a JSON-serialisable dict.
* Never raises. Failures are {"ok": False, "error": {"code", "message", "details"}}.
* Output is compact (the LLM reads it): no raw attribute dumps, bounded lists.
* The LLM never chooses file paths; subsets are written under ctx.output_dir.
* Every call is logged (provenance) with its arguments and outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time as _time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import subsetting as sub
import verification as ver
from catalog_adapters import CatalogAdapter, _lon_intervals, _time_bound


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------
@dataclass
class ToolContext:
    catalog: CatalogAdapter
    output_dir: str | Path = "subsets"
    default_max_bytes: int = 500_000_000
    provenance_path: str | Path | None = None   # JSONL, appended to on every call
    today: date | None = None                   # fixed "today" for tests; default: real date
    require_estimate: bool = True               # subset_dataset must repeat an estimated request
    estimated: dict[str, dict] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)
    subsets: dict[str, dict] = field(default_factory=dict)
    share_remote_handles: bool = True           # open each remote source once per run
    _open: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def open_source(self, source: str, fmt: str | None):
        """The opened dataset for `source`. Remote sources are opened once and reused until
        close(): estimate and subset would otherwise each hit the server again, and public
        servers (ERDDAP, THREDDS) rate-limit. Datasets open lazily, so a handle holds only
        metadata and the 1-D coordinates, whatever the size of the data. Local sources are
        returned as a path (opened and closed per call, as before). Raises SubsetError."""
        if not self.share_remote_handles or "://" not in str(source):
            return source
        key = f"{fmt}|{source}"
        if key not in self._open:                    # a failed open is never cached
            self._open[key] = sub.open_source(source, fmt)
        return self._open[key]

    def close(self) -> None:
        """Release every shared handle. Called by the agent at the end of a run."""
        handles, self._open = list(self._open.values()), {}
        for ds in handles:
            try:
                ds.close()
            except Exception:  # noqa: BLE001  (closing must never mask the run's result)
                pass


def err(code: str, message: str, **details: Any) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "details": details}}


# --------------------------------------------------------------------------
# Argument validation with light coercion (small LLMs get types wrong often)
# --------------------------------------------------------------------------
def _coerce(schema: dict, value: Any, path: str, errors: list[str]) -> Any:
    t = schema.get("type")

    if t == "object":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if not isinstance(value, dict):
            errors.append(f"{path}: expected an object, got {type(value).__name__}")
            return value
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        extra = schema.get("additionalProperties", False)
        out: dict = {}
        for k, v in value.items():
            if v is None and k not in required:
                continue  # null for an optional field == omitted
            if k in props:
                out[k] = _coerce(props[k], v, f"{path}.{k}", errors)
            elif isinstance(extra, dict):
                out[k] = _coerce(extra, v, f"{path}.{k}", errors)
            elif extra is True:
                out[k] = v
            else:
                hint = " Pass them directly as top-level arguments." if path == "arguments" else ""
                errors.append(f"{path}: unexpected property '{k}' "
                              f"(allowed: {sorted(props)}).{hint}")
        for k, p in props.items():
            if k not in out and "default" in p:
                out[k] = p["default"]
        for k in required:
            if k not in out:
                errors.append(f"{path}: missing required property '{k}'")
        return out

    if t == "array":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = [value] if schema.get("items", {}).get("type") == "string" else value
        if not isinstance(value, list):
            errors.append(f"{path}: expected an array, got {type(value).__name__}")
            return value
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        return [_coerce(schema.get("items", {}), v, f"{path}[{i}]", errors)
                for i, v in enumerate(value)]

    if t == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            errors.append(f"{path}: expected a string, got {type(value).__name__}")
            return value
        if "enum" in schema:
            match = {e.lower(): e for e in schema["enum"]}.get(value.strip().lower())
            if match is None:
                errors.append(f"{path}: must be one of {schema['enum']}, got {value!r}")
                return value
            return match
        return value

    if t in ("integer", "number"):
        v = value
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                errors.append(f"{path}: expected a {t}, got {value!r}")
                return value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{path}: expected a {t}, got {type(value).__name__}")
            return value
        if t == "integer":
            if isinstance(v, float) and not v.is_integer():
                errors.append(f"{path}: expected an integer, got {value!r}")
                return value
            v = int(v)
        if "minimum" in schema and v < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}, got {v}")
        if "maximum" in schema and v > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}, got {v}")
        return v

    if t == "boolean":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        if not isinstance(value, bool):
            errors.append(f"{path}: expected true/false, got {value!r}")
        return value

    return value


# --------------------------------------------------------------------------
# Tool + registry
# --------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                      # JSON Schema (type: object)
    func: Callable[..., dict]             # func(ctx, **validated_args) -> dict


class ToolRegistry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # ---- schemas for the two LLM backends ----
    def anthropic_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.parameters} for t in self._tools.values()]

    def ollama_tools(self) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in self._tools.values()]

    # ---- execution ----
    def call(self, name: str, arguments: dict | str | None = None) -> dict:
        t0 = _time.perf_counter()
        raw_args = arguments
        result: dict
        try:
            tool = self._tools.get(name)
            if tool is None:
                result = err("unknown_tool", f"No tool named '{name}'.",
                             available=self.names)
            else:
                if arguments is None:
                    arguments = {}
                problems: list[str] = []
                args = _coerce(tool.parameters, arguments, "arguments", problems)
                if problems:
                    result = err("invalid_arguments",
                                 "; ".join(problems),
                                 problems=problems, expected_parameters=tool.parameters)
                else:
                    result = tool.func(self.ctx, **args)
                    raw_args = args  # log the coerced arguments
        except Exception as e:  # noqa: BLE001  (a tool must never crash the agent loop)
            result = err("tool_crashed", f"{type(e).__name__}: {e}",
                         traceback=traceback.format_exc(limit=3))
        self._log(name, raw_args, result, _time.perf_counter() - t0)
        return result

    def _log(self, name: str, args: Any, result: dict, seconds: float) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": name,
            "arguments": args,
            "ok": bool(result.get("ok")),
            "error_code": (result.get("error") or {}).get("code"),
            "summary": {k: result[k] for k in ("subset_id", "n_results", "output_bytes",
                                               "dataset_id") if k in result},
            "duration_s": round(seconds, 3),
        }
        self.ctx.log.append(entry)
        if self.ctx.provenance_path:
            with open(self.ctx.provenance_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")


def to_text(result: dict, max_chars: int = 8000) -> str:
    """Serialise a tool result for the LLM, with a hard length cap."""
    s = json.dumps(result, ensure_ascii=False, default=str)
    return s if len(s) <= max_chars else s[:max_chars] + f"... [truncated {len(s) - max_chars} chars]"


# --------------------------------------------------------------------------
# Shared schema fragments
# --------------------------------------------------------------------------
BBOX_SCHEMA = {
    "type": "object",
    "description": ("Geographic box in degrees. Longitudes in degrees east, either "
                    "-180..180 or 0..360. The box runs west to east from lon_min to "
                    "lon_max; if the region crosses the antimeridian use lon_min > lon_max "
                    "(e.g. 170 and -170)."),
    "properties": {
        "lat_min": {"type": "number", "minimum": -90, "maximum": 90},
        "lat_max": {"type": "number", "minimum": -90, "maximum": 90},
        "lon_min": {"type": "number", "minimum": -360, "maximum": 360},
        "lon_max": {"type": "number", "minimum": -360, "maximum": 360},
    },
    "required": ["lat_min", "lat_max", "lon_min", "lon_max"],
}

TIME_SCHEMA = {
    "type": "object",
    "description": ("Time range as ISO dates (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). "
                    "Omit 'start' or 'end' for an open-ended range. A date-only 'end' "
                    "includes that whole day."),
    "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
}

VARIABLES_SCHEMA = {
    "type": "array", "items": {"type": "string"}, "minItems": 1,
    "description": "Exact variable names as listed by get_dataset (e.g. ['sst']).",
}

STRIDE_SCHEMA = {
    "type": "object",
    "description": ("Optional thinning: keep every n-th point. Keys are 'lat', 'lon', "
                    "'time' or a dimension name; values are integers >= 1. "
                    "Example: {\"lat\": 4, \"lon\": 4} gives a 4x coarser grid."),
    "additionalProperties": {"type": "integer", "minimum": 1},
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _bbox_tuple(b: dict | None):
    return None if not b else (b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"])


def _time_tuple(t: dict | None):
    if not t or (t.get("start") is None and t.get("end") is None):
        return None
    return (t.get("start"), t.get("end"))


def _build_request(variables, bbox, time, stride) -> dict:
    req: dict[str, Any] = {"variables": variables}
    if bbox:
        req["bbox"] = _bbox_tuple(bbox)
    if _time_tuple(time):
        req["time"] = _time_tuple(time)
    if stride:
        req["stride"] = stride
    return req


def _request_key(dataset_id: str, req: dict) -> str:
    """Canonical identity of a subset request (size limit and output format excluded)."""
    return json.dumps({"d": dataset_id, "v": sorted(req.get("variables") or []),
                       "b": req.get("bbox"), "t": req.get("time"), "s": req.get("stride") or {}},
                      sort_keys=True, default=str)


def _canon(value: Any) -> str:
    return json.dumps(value or None, sort_keys=True, default=str)


def _not_estimated(ctx: ToolContext, dataset_id: str, req: dict) -> dict:
    """subset_dataset was called with arguments that were never size-checked."""
    previous = [e for e in ctx.estimated.values() if e["dataset_id"] == dataset_id]
    if not previous:
        return err("not_estimated", "Call estimate_subset_size with these arguments first; "
                   "subset_dataset only accepts a request that was size-checked.")
    last = previous[-1]["request"]
    diffs = []
    for k in ("variables", "bbox", "time", "stride"):
        was, now = last.get(k), req.get(k)
        if k == "variables":
            was, now = sorted(was or []), sorted(now or [])
        if _canon(was) != _canon(now):
            diffs.append(f"{k}: estimated {was or 'not set'}, this call {now or 'not set'}")
    return err("not_estimated",
               "These arguments differ from your last estimate_subset_size for this dataset: "
               + "; ".join(diffs) + ". Use exactly the arguments you estimated, or estimate the "
               "new request first.", differences=diffs, last_estimated_request=last)


def _resolve(ctx: ToolContext, dataset_id: str):
    rec = ctx.catalog.get(dataset_id)
    if rec is None:
        return None, err("unknown_dataset", f"No dataset with id '{dataset_id}'. "
                         "Use search_catalog to find valid dataset_ids.")
    return rec, None


def _trim(x: Any, n: int = 300) -> Any:
    return x[:n] + "..." if isinstance(x, str) and len(x) > n else x


_ATTR_KEYS = ("title", "summary", "description", "institution", "source", "references",
              "license", "comment", "Conventions", "version", "product_version",
              "processing_level")


def _var_summary(v: dict) -> dict:
    a = v.get("attributes") or {}
    out: dict[str, Any] = {
        "name": v["name"], "long_name": a.get("long_name"),
        "standard_name": a.get("standard_name"), "units": a.get("units"),
        "dims": v.get("dimensions"), "shape": v.get("shape"), "dtype": v.get("dtype"),
    }
    ch = v.get("chunks")
    if ch:
        out["chunk_shape"] = [c[0] if isinstance(c, list) and c else c for c in ch]
    st = v.get("statistics")
    if st:
        n = (st.get("count") or 0) + (st.get("nan_count") or 0)
        out["stats"] = {"min": st.get("min"), "max": st.get("max"), "mean": st.get("mean"),
                        "nan_fraction": round(st["nan_count"] / n, 4)
                        if n and st.get("nan_count") is not None else None}
    return {k: val for k, val in out.items() if val is not None}


def _search_hit(rec: dict) -> dict:
    attrs = rec.get("global_attributes") or {}
    vars_ = rec.get("variables", [])
    labels = []
    for v in vars_[:8]:
        a = v.get("attributes") or {}
        extra = ", ".join(str(x) for x in (a.get("long_name"), a.get("units")) if x)
        labels.append(f"{v['name']} ({extra})" if extra else v["name"])
    return {k: val for k, val in {
        "dataset_id": rec["dataset_id"], "title": _trim(attrs.get("title"), 120),
        "score": rec.get("score"), "matched_via": rec.get("matched_via"),
        "format": rec.get("format"), "spatial": rec.get("spatial"),
        "temporal": rec.get("temporal"), "n_variables": len(vars_),
        "variables": labels,
    }.items() if val is not None}


def _clean_error(res: dict) -> dict:
    """Trim bulky error details before they reach the LLM."""
    e = res.get("error", {})
    d = dict(e.get("details") or {})
    if "estimate" in d:
        est = d.pop("estimate")
        d["estimate"] = {"shape": est.get("shape"), "output_size": est.get("output_human")}
    for k in ("available", "attempts", "missing"):
        if isinstance(d.get(k), list):
            d[k] = d[k][:30]
    return err(e.get("code", "error"), e.get("message", ""), **d)


def _size_hints(est: dict, max_bytes: int) -> list[str]:
    if est["output_bytes"] <= max_bytes:
        return []
    factor = est["output_bytes"] / max_bytes
    shape = est["shape"]
    two = sorted(shape, key=shape.get, reverse=True)[:2]   # spread the thinning
    per_dim = math.ceil(factor ** (1 / len(two)))
    example = {d: min(per_dim, shape[d]) for d in two}
    hints = [f"Output is {factor:.1f}x over the limit; reduce it by at least that factor.",
             "Options: a smaller region, a shorter period, or thinning the largest "
             f"dimensions, e.g. stride {json.dumps(example)}."]
    if len(est["variables"]) > 1:
        hints.append("Requesting fewer variables also reduces the size.")
    return hints


# --------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------
def search_catalog(ctx: ToolContext, query=None, format=None, bbox=None, time=None,
                   limit=5) -> dict:
    bb, tt = _bbox_tuple(bbox), _time_tuple(time)
    try:
        hits = ctx.catalog.search(query, format=format, bbox=bb, time=tt, limit=limit)
    except Exception as e:  # noqa: BLE001
        return err("catalog_error", f"{type(e).__name__}: {e}")
    out = {
        "ok": True, "n_results": len(hits),
        "applied_filters": {"query": query, "format": format, "bbox": bbox, "time": time},
        "results": [_search_hit(h) for h in hits],
    }
    if not hits:
        out["hint"] = ("No dataset matched. Remove filters one at a time (applied_filters "
                       "shows what was active) or use more general query words.")
    return out


def get_dataset(ctx: ToolContext, dataset_id: str, variables=None,
                include_attributes=False) -> dict:
    rec, e = _resolve(ctx, dataset_id)
    if e:
        return e
    attrs = rec.get("global_attributes") or {}
    all_vars = rec.get("variables", [])
    chosen = [v for v in all_vars if not variables or v["name"] in set(variables)]
    notes = []
    if variables:
        missing = sorted(set(variables) - {v["name"] for v in all_vars})
        if missing:
            notes.append(f"Unknown variables: {missing}. "
                         f"Available: {[v['name'] for v in all_vars][:30]}")
    if len(chosen) > 40:
        notes.append(f"Showing 40 of {len(chosen)} variables; pass `variables` to see others.")
        chosen = chosen[:40]
    out: dict[str, Any] = {
        "ok": True, "dataset_id": dataset_id,
        "title": attrs.get("title"), "description": _trim(
            attrs.get("summary") or attrs.get("description")),
        "institution": attrs.get("institution"),
        "format": rec.get("format"), "is_remote": rec.get("is_remote"),
        "spatial": rec.get("spatial"), "temporal": rec.get("temporal"),
        "dimensions": rec.get("dimensions"),
        "variables": [_var_summary(v) for v in chosen],
        "warnings": (rec.get("warnings") or [])[:5],
    }
    if include_attributes:   # the informative ones only; the rest costs tokens on every later call
        by_lower = {k.lower(): k for k in attrs}
        out["global_attributes"] = {
            by_lower[k.lower()]: _trim(attrs[by_lower[k.lower()]], 200) for k in _ATTR_KEYS
            if k.lower() in by_lower and attrs[by_lower[k.lower()]] not in (None, "")}
    if notes:
        out["notes"] = notes
    return {k: v for k, v in out.items() if v not in (None, [], {})} | {"ok": True}


def estimate_subset_size(ctx: ToolContext, dataset_id: str, variables: list,
                         bbox=None, time=None, stride=None, max_bytes=None) -> dict:
    rec, e = _resolve(ctx, dataset_id)
    if e:
        return e
    limit = max_bytes or ctx.default_max_bytes
    request = _build_request(variables, bbox, time, stride)
    try:
        src = ctx.open_source(rec["source"], rec.get("format"))
    except sub.SubsetError as ex:
        return _clean_error(ex.to_dict())
    res = sub.estimate_subset_size(src, request, max_bytes=limit,
                                   source_format=rec.get("format"))
    if not res["ok"]:
        return _clean_error(res)
    key = _request_key(dataset_id, request)
    ctx.estimated.pop(key, None)   # re-insert: the most recent estimate stays last
    ctx.estimated[key] = {"dataset_id": dataset_id, "request": request, "shape": res["shape"]}
    out = {
        "ok": True, "dataset_id": dataset_id, "shape": res["shape"],
        "output_size": res["output_human"], "output_bytes": res["output_bytes"],
        "data_read": res["read_human"], "read_amplification": res["read_amplification"],
        "max_bytes": limit, "within_limit": res["within_limit"],
        "warnings": res["warnings"],
    }
    if res["within_limit"]:   # subset_dataset's own next_step is followed reliably; give the estimate one
        out["next_step"] = ("Nothing has been extracted yet. To extract, call subset_dataset with "
                            "exactly these arguments (variables, bbox, time, stride), then verify_subset.")
    hints = _size_hints(res, limit)
    if res["read_amplification"] and res["read_amplification"] > 20:
        hints.append("The source chunk layout makes this request read far more data than "
                     "it keeps; this is slow but does not change the output size.")
    if hints:
        out["hints"] = hints
    return out


def subset_dataset(ctx: ToolContext, dataset_id: str, variables: list, bbox=None,
                   time=None, stride=None, output_format="zarr", max_bytes=None) -> dict:
    rec, e = _resolve(ctx, dataset_id)
    if e:
        return e
    limit = max_bytes or ctx.default_max_bytes
    request = _build_request(variables, bbox, time, stride)
    if ctx.require_estimate and _request_key(dataset_id, request) not in ctx.estimated:
        return _not_estimated(ctx, dataset_id, request)
    key = json.dumps({"d": dataset_id, "r": request, "f": output_format},
                     sort_keys=True, default=str)
    subset_id = "sub_" + hashlib.sha256(key.encode()).hexdigest()[:10]
    ext = ".zarr" if output_format == "zarr" else ".nc"
    out_path = Path(ctx.output_dir) / f"{subset_id}{ext}"

    try:
        src = ctx.open_source(rec["source"], rec.get("format"))
    except sub.SubsetError as ex:
        return _clean_error(ex.to_dict())
    res = sub.subset_dataset(src, request, out_path, fmt=output_format,
                             max_bytes=limit, overwrite=True,
                             source_format=rec.get("format"), source_label=rec["source"])
    if not res["ok"]:
        return _clean_error(res)

    record = {
        "subset_id": subset_id, "dataset_id": dataset_id, "source": rec["source"],
        "request": res["request"], "output_path": res["output_path"],
        "output_format": output_format, "shape": res["shape"],
        "variables": res["variables"], "file_bytes": res["file_bytes"], "extent": res["extent"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": res["warnings"],
    }
    ctx.subsets[subset_id] = record
    (Path(ctx.output_dir) / f"{subset_id}.provenance.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    return {"ok": True, "subset_id": subset_id, "dataset_id": dataset_id,
            "output_path": res["output_path"], "shape": res["shape"],
            "extent": res["extent"],
            "variables": res["variables"], "output_bytes": res["file_bytes"],
            "warnings": res["warnings"],
            "next_step": "Call verify_subset with this subset_id."}


def verify_subset(ctx: ToolContext, subset_id: str) -> dict:
    record = ctx.subsets.get(subset_id)
    if record is None:  # e.g. a new session: reload from the sidecar file
        side = Path(ctx.output_dir) / f"{subset_id}.provenance.json"
        if side.exists():
            record = json.loads(side.read_text(encoding="utf-8"))
            ctx.subsets[subset_id] = record
    if record is None:
        return err("unknown_subset", f"No subset with id '{subset_id}'.",
                   known=list(ctx.subsets)[:20])

    rec = ctx.catalog.get(record["dataset_id"]) or {}
    wanted = set(record["variables"])
    ref, units = {}, {}
    for v in rec.get("variables", []):
        if v["name"] not in wanted:
            continue
        st = v.get("statistics") or {}
        if st.get("min") is not None and st.get("max") is not None:
            ref[v["name"]] = {"min": st["min"], "max": st["max"]}
        u = (v.get("attributes") or {}).get("units")
        if u:
            units[v["name"]] = u

    res = ver.verify_subset(record["output_path"], record["request"],
                            reference_stats=ref, expected_units=units)
    summary = {
        name: {k: info[k] for k in ("min", "max", "mean", "nan_fraction", "units")
               if info.get(k) is not None}
        for name, info in res["summary"].items()
    }
    return {"ok": True, "subset_id": subset_id, "verified": res["ok"],
            "n_errors": res["n_errors"], "n_warnings": res["n_warnings"],
            "issues": res["issues"][:20], "variables": summary}


def resolve_time_range(ctx: ToolContext, year=None, last_amount=None, last_unit=None) -> dict:
    """Date arithmetic in code, not in the LLM."""
    today = ctx.today or date.today()
    if year is not None and last_amount is None and last_unit is None:
        start, end = f"{year:04d}-01-01", f"{year:04d}-12-31"
    elif last_amount is not None and last_unit is not None and year is None:
        end_ts = pd.Timestamp(today)
        start_ts = end_ts - pd.DateOffset(**{last_unit: last_amount})
        start, end = start_ts.date().isoformat(), end_ts.date().isoformat()
    else:
        return err("bad_request", "Give either `year`, or both `last_amount` and `last_unit`.")
    return {"ok": True, "start": start, "end": end, "today": today.isoformat()}



# --------------------------------------------------------------------------
# compare_datasets: pick the best candidate for ONE task, deterministically
# --------------------------------------------------------------------------
_SENTINELS = (-999.0, -999.9, -9999.0, -1000.0, -32768.0, -32767.0,
              1e20, 1e30, -1e30, 9.96921e36, 1e36)
_LAT_DIMS = {"lat", "latitude", "y", "nav_lat", "rlat", "grid_latitude"}
_LON_DIMS = {"lon", "longitude", "x", "nav_lon", "rlon", "grid_longitude"}


def _suspect_fill(st: dict | None) -> list[str]:
    """Catalog min/max that look like an unmasked fill value. A warning, not a verdict:
    -1000 is also a legitimate bathymetry value."""
    out = []
    for k in ("min", "max"):
        v = (st or {}).get(k)
        if v is None:
            continue
        if abs(v) >= 1e10 or any(math.isclose(v, s, rel_tol=1e-4) for s in _SENTINELS):
            out.append(f"{k}={v:g}")
    return out


def _ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_convert(None) if t.tzinfo else t


def _time_cover(tp: dict, time) -> float | None:
    """Fraction of the requested period that lies inside the dataset period."""
    if not time or (time[0] is None and time[1] is None) or not tp.get("start") or not tp.get("end"):
        return None
    d0, d1 = _ts(tp["start"]), _ts(tp["end"])
    r0 = _ts(_time_bound(time[0], False)) if time[0] else d0
    r1 = _ts(_time_bound(time[1], True)) if time[1] else d1
    if r1 <= r0:
        return 1.0 if d0 <= r0 <= d1 else 0.0
    return round(max(0.0, (min(r1, d1) - max(r0, d0)) / (r1 - r0)), 3)


def _space_cover(sp: dict, bbox) -> float | None:
    """Fraction of the requested box that lies inside the dataset extent."""
    if not bbox or any(sp.get(k) is None for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
        return None
    lat_min, lat_max, lon_min, lon_max = bbox
    if lat_max > lat_min:
        lat_f = max(0.0, min(lat_max, sp["lat_max"]) - max(lat_min, sp["lat_min"])) / (lat_max - lat_min)
    else:
        lat_f = 1.0 if sp["lat_min"] <= lat_min <= sp["lat_max"] else 0.0
    total = inside = 0.0
    for a, b in _lon_intervals(lon_min, lon_max):
        total += b - a
        inside += max(0.0, min(b, sp["lon_max"]) - max(a, sp["lon_min"]))
    lon_f = inside / total if total > 0 else (
        1.0 if any(sp["lon_min"] <= a <= sp["lon_max"] for a, _ in _lon_intervals(lon_min, lon_max)) else 0.0)
    return round(lat_f * lon_f, 3)


def _norm(s: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s or "").lower()).strip()


def _match_score(v: dict, term: str) -> int:
    """3 = the term is the variable name or its standard_name; 2 = all its words are in
    long_name/standard_name; 1 = it appears inside name/long_name/standard_name; 0 = no match."""
    t = _norm(term)
    a = v.get("attributes") or {}
    name, std, long_ = _norm(v["name"]), _norm(a.get("standard_name")), _norm(a.get("long_name"))
    if t and t in (name, std):
        return 3
    words = set(t.split())
    if words and (words <= set(std.split()) or words <= set(long_.split())):
        return 2
    return 1 if t and t in f"{name} {std} {long_}" else 0


def _grid_info(v: dict, sp: dict) -> dict:
    """Is the variable on a regular lat/lon grid (so bbox subsetting can work), and how fine."""
    dims = [str(d) for d in (v.get("dimensions") or [])]
    lat_d = next((d for d in dims if d.lower() in _LAT_DIMS), None)
    lon_d = next((d for d in dims if d.lower() in _LON_DIMS), None)
    info: dict[str, Any] = {"gridded": bool(lat_d and lon_d), "dims": dims}
    shape = dict(zip(dims, v.get("shape") or []))
    for key, d in (("lat", lat_d), ("lon", lon_d)):
        n = shape.get(d)
        if d and n and n > 1 and sp.get(f"{key}_min") is not None:
            info[f"{key}_step_deg"] = round((sp[f"{key}_max"] - sp[f"{key}_min"]) / (n - 1), 4)
    return info


def compare_datasets(ctx: ToolContext, dataset_ids: list, variable=None, bbox=None, time=None) -> dict:
    if len(set(dataset_ids)) < 2 or len(set(dataset_ids)) > 4:
        return err("bad_request", "Give between 2 and 4 different dataset_ids.")
    bb, tt = _bbox_tuple(bbox), _time_tuple(time)
    rows, unknown = [], []
    for did in dict.fromkeys(dataset_ids):
        rec = ctx.catalog.get(did)
        if rec is None:
            unknown.append(did)
            continue
        sp, tp = rec.get("spatial") or {}, rec.get("temporal") or {}
        flags: list[str] = []
        eligible = True
        row: dict[str, Any] = {
            "dataset_id": did,
            "title": _trim((rec.get("global_attributes") or {}).get("title"), 100),
            "format": rec.get("format"),
            "period": [tp.get("start"), tp.get("end")],
            "extent": sp or None,
            "coverage": {"space": _space_cover(sp, bb), "time": _time_cover(tp, tt)},
        }
        if variable:
            ranked = sorted(((_match_score(v, variable), v) for v in rec.get("variables", [])),
                            key=lambda x: -x[0])
            ranked = [(s_, v) for s_, v in ranked if s_ > 0]
            if not ranked:
                flags.append(f"no variable matches '{variable}'")
                eligible = False
            else:
                best = ranked[0][1]
                s_ = _var_summary(best)
                row["variable"] = {k: s_[k] for k in ("name", "long_name", "standard_name", "units",
                                                      "shape", "stats") if k in s_}
                row["match_strength"] = ranked[0][0]
                grid = _grid_info(best, sp)
                row["grid"] = grid
                if len([1 for x, _ in ranked if x == ranked[0][0]]) > 1:
                    flags.append(f"several variables match equally well; showing '{best['name']}'")
                if bb and not grid["gridded"]:
                    flags.append("variable is not on a regular lat/lon grid (dims: "
                                 f"{grid['dims']}); bbox subsetting is likely to be rejected "
                                 "(unsupported_grid)")
                    eligible = False
                bad = _suspect_fill(best.get("statistics"))
                if bad:
                    flags.append("catalog statistics look like an unmasked fill value ("
                                 + ", ".join(bad) + "); verify_subset's range check is unreliable here")
                nf = (row["variable"].get("stats") or {}).get("nan_fraction")
                if nf is not None and nf > 0.5:
                    flags.append(f"{nf:.0%} of values are NaN")
        for k, label in (("space", "region"), ("time", "period")):
            c = row["coverage"][k]
            if c is not None and c == 0:
                flags.append(f"does not cover the requested {label} at all")
                eligible = False
            elif c is not None and c < 1:
                flags.append(f"covers only {'<1%' if c < 0.005 else format(c, '.0%')} of the requested {label}")
        if eligible is False:
            row["eligible"] = False
        if flags:
            row["flags"] = flags
        row["_score"] = round((row["coverage"]["space"] if row["coverage"]["space"] is not None else 1.0)
                              * (row["coverage"]["time"] if row["coverage"]["time"] is not None else 1.0), 3)
        row["_eligible"] = eligible
        rows.append(row)
    if not rows:
        return err("unknown_dataset", "None of the dataset_ids exist.", unknown=unknown)

    # eligible first, then coverage, then strength of the variable match, then fewer flags
    rows.sort(key=lambda r: (not r["_eligible"], -r["_score"], -(r.get("match_strength") or 0),
                             len(r.get("flags", []))))
    good = [r for r in rows if r["_eligible"]]
    out: dict[str, Any] = {"ok": True, "n": len(rows)}
    if not good:
        out["recommended"] = None
        out["recommendation"] = ("No candidate fits: each fails on variable, grid or coverage (see flags). "
                                 "Relax the request or search again; do not extract a poor match "
                                 "without telling the user.")
    else:
        top = good[0]
        out["recommended"] = top["dataset_id"]
        close = len(good) > 1 and (top["_score"] - good[1]["_score"]) < 0.1
        out["confidence"] = "close" if close else "clear"
        why = [f"covers {top['_score']:.0%} of the requested region x period"]
        if len(good) > 1:
            why.append(f"next best covers {good[1]['_score']:.0%}")
        out["recommendation"] = "; ".join(why) + "."
        if close:
            out["recommendation"] += (" Candidates are close on coverage: decide on grid spacing, "
                                      "units and flags against what the user asked for.")
    for r in rows:
        r.pop("_score"), r.pop("_eligible")
    out["datasets"] = rows
    out["ranking_basis"] = ("fit first (variable matches, regular lat/lon grid when a region is "
                            "requested, some coverage), then coverage of region x period. Grid spacing "
                            "and units are shown but not ranked: judge them against the task.")
    if unknown:
        out["unknown_dataset_ids"] = unknown
    return out


# --------------------------------------------------------------------------
# Registry assembly
# --------------------------------------------------------------------------
def build_registry(ctx: ToolContext) -> ToolRegistry:
    reg = ToolRegistry(ctx)
    ds_id = {"type": "string", "description": "A dataset_id returned by search_catalog."}

    reg.register(Tool(
        "resolve_time_range",
        "Convert a calendar year ('2019') or a relative period ('the last 5 years') into "
        "exact start and end dates for the `time` filters. Always use this instead of "
        "working out dates yourself.",
        {"type": "object", "properties": {
            "year": {"type": "integer", "minimum": 1800, "maximum": 2200},
            "last_amount": {"type": "integer", "minimum": 1,
                            "description": "With last_unit: 'the last N units' up to today."},
            "last_unit": {"type": "string", "enum": ["days", "months", "years"]},
        }},
        resolve_time_range))

    reg.register(Tool(
        "search_catalog",
        "Find datasets in the catalog. 'query' should contain only the topic or variable "
        "you are looking for (e.g. 'sea surface temperature'); give the region as `bbox`, the "
        "dates as `time` and the file type as `format`, never inside the query. Datasets are matched by "
        "what they describe AND by the variables they contain. Returns compact summaries.",
        {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "Topic or variable words only, no region/dates/format."},
            "format": {"type": "string", "enum": ["netcdf", "zarr"]},
            "bbox": BBOX_SCHEMA, "time": TIME_SCHEMA,
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        }},
        search_catalog))

    reg.register(Tool(
        "get_dataset",
        "Get details of one dataset: description, spatial and temporal extent, dimensions, "
        "and per-variable units, shape, chunking and statistics (min, max, mean, NaN "
        "fraction). Use it to compare candidates before subsetting.",
        {"type": "object", "properties": {
            "dataset_id": ds_id,
            "variables": {"type": "array", "items": {"type": "string"},
                          "description": "Only show these variables."},
            "include_attributes": {"type": "boolean", "default": False,
                                   "description": "Also return the main global attributes "
                                   "(title, summary, institution, references, license, comment)."},
        }, "required": ["dataset_id"]},
        get_dataset))

    reg.register(Tool(
        "estimate_subset_size",
        "Check a planned subset BEFORE extracting it: returns the output size, how much "
        "source data would be read, and whether it fits the size limit. Also fails with a "
        "clear reason if the region or period does not overlap the dataset. Reads no data. "
        "subset_dataset only accepts a request that was estimated first.",
        {"type": "object", "properties": {
            "dataset_id": ds_id, "variables": VARIABLES_SCHEMA, "bbox": BBOX_SCHEMA,
            "time": TIME_SCHEMA, "stride": STRIDE_SCHEMA,
            "max_bytes": {"type": "integer", "minimum": 1,
                          "description": "Size limit in bytes (default: system limit)."},
        }, "required": ["dataset_id", "variables"]},
        estimate_subset_size))

    reg.register(Tool(
        "subset_dataset",
        "Extract variables, a geographic box and a time range from a dataset into a new "
        "file. Refuses if the result would exceed the size limit. Returns a subset_id; "
        "always call verify_subset on it afterwards. Use EXACTLY the arguments you passed to "
        "estimate_subset_size (same variables, bbox, time and stride); different arguments "
        "are refused.",
        {"type": "object", "properties": {
            "dataset_id": ds_id, "variables": VARIABLES_SCHEMA, "bbox": BBOX_SCHEMA,
            "time": TIME_SCHEMA, "stride": STRIDE_SCHEMA,
            "output_format": {"type": "string", "enum": ["zarr", "netcdf"], "default": "zarr"},
            "max_bytes": {"type": "integer", "minimum": 1},
        }, "required": ["dataset_id", "variables"]},
        subset_dataset))

    reg.register(Tool(
        "verify_subset",
        "Check an extracted subset: is it empty, all NaN, outside the requested region or "
        "period, in unexpected units, or outside the value range known for the dataset? "
        "Returns issues with severity 'error' or 'warning'. 'verified: false' means the "
        "subset is not trustworthy.",
        {"type": "object", "properties": {
            "subset_id": {"type": "string",
                          "description": "The subset_id returned by subset_dataset."},
        }, "required": ["subset_id"]},
        verify_subset))

    reg.register(Tool(
        "compare_datasets",
        "Compare 2 to 4 candidate datasets for the same task and get a recommendation: how much of "
        "the requested region and period each covers, the matching variable with units, grid "
        "spacing, NaN fraction, whether it is on a regular lat/lon grid, and flags (partial "
        "coverage, suspicious statistics). Returns `recommended` (a dataset_id or null) with the "
        "reason. Use it after search_catalog when there are 2 or more candidates.",
        {"type": "object", "properties": {
            "dataset_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                            "description": "dataset_ids returned by search_catalog."},
            "variable": {"type": "string", "description": "Topic or variable name, e.g. 'sea surface temperature'."},
            "bbox": BBOX_SCHEMA, "time": TIME_SCHEMA,
        }, "required": ["dataset_ids"]},
        compare_datasets))
    return reg
