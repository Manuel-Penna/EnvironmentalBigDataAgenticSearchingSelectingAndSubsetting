# GeoMeta

GeoMeta is a catalog of NetCDF and Zarr datasets with an LLM agent on top.

- **The catalog** extracts metadata, per-variable statistics and embeddings from NetCDF and Zarr sources (local files, HTTP, OPeNDAP, cloud object stores), stores them in [Qdrant](https://qdrant.tech/), and exposes them through a REST API and a small web interface.
- **The agent** takes a task in plain language ("sea surface temperature around Naples in 2026"), finds a suitable dataset in the catalog, extracts exactly the requested subset (variables, region, period), and checks the result.

The LLM decides *which tool to call and with what arguments*. Code does everything deterministic: dates, bounding boxes, size estimates, subsetting, verification, and the workflow rules.

This repository has two entry points:

| Part | Entry point | What it does |
|---|---|---|
| Catalog | `python run.py` | Flask app: ingest datasets, browse the catalog, structured and one-shot natural-language search |
| Agent | `python run_agent.py "<task>"` | Command-line agent that searches the catalog and extracts a verified subset |

The agent works only on datasets that are already in the catalog, so you ingest first (Part 1), then use the agent (Part 2).

---

## Contents

- [Architecture](#architecture)
- [Installation (Linux)](#installation-linux)
- [Configuration](#configuration)
- [Part 1: run the catalog (`run.py`)](#part-1-run-the-catalog-runpy)
- [Part 2: run the agent (`run_agent.py`)](#part-2-run-the-agent-run_agentpy)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)

---

## Architecture

```
                       ┌───────────────────────────────┐
  Catalog (Part 1)     │  run.py  →  Flask app         │
                       │  /datasets/*  /query/*  /     │
                       └──────────────┬────────────────┘
                                      │ extract / query
        extractors.py ── embeddings.py ── qdrant_repository.py ── qdrant_query_engine.py
        (NetCDF, Zarr)   (Ollama)         (Qdrant, 2 point types)   (filters + semantic search)
                                      │
                                      ▼
                                   Qdrant  ◄──────────────────────────────┐
                                                                          │
  Agent (Part 2)       run_agent.py                                       │
                          │                                               │
                          ▼                                               │
                     agent.py  ── LLM: Groq or Ollama (via LangChain)     │
                          │   think → call tool → read result → repeat    │
                          ▼                                               │
                  agent_tools.py  (validation, structured errors, provenance)
                     resolve_time_range   search_catalog ─ catalog_adapters.py ┘
                     compare_datasets     get_dataset
                     estimate_subset_size subset_dataset ─ subsetting.py ─ extractors.py openers
                     verify_subset ───────────────────────  verification.py
```

Qdrant stores two kinds of points per dataset: one for the dataset description and one per variable (name, long name, standard name, units, statistics). Semantic search runs in two stages (datasets, then variables) and is combined with structured pre-filters (format, bounding box, time range).

A UML class diagram is in [`geometa_uml.puml`](geometa_uml.puml).

---

## Installation (Linux)

### 1. System requirements

- Python 3.10 or newer
- Docker (for Qdrant), or a Qdrant binary
- [Ollama](https://ollama.com/) (needed for embeddings in both parts, and optionally as the LLM)
- Optional: a [Groq](https://console.groq.com/) API key for the agent, an Anthropic API key for the one-shot natural-language endpoint

### 2. Get the code and a virtual environment

```bash
git clone <your-repo-url> geometa
cd geometa

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Python packages

There is no pinned requirements file yet, so install the packages by group.

```bash
# Catalog: web app, storage, embeddings, data access
pip install flask qdrant-client ollama python-dateutil \
            numpy pandas xarray dask zarr fsspec aiohttp \
            netCDF4 h5netcdf scipy

# OPeNDAP sources (dap2:// / dap4:// URLs)
pip install pydap

# Agent
pip install langchain-core langchain-ollama langchain-groq python-dotenv

# One-shot natural-language endpoint with Claude (optional)
pip install anthropic

# Tests
pip install pytest
```

> **Qdrant client version.** Recent `qdrant-client` releases removed `QdrantClient.search()`. If you see an `AttributeError` mentioning `search`, install an older release (`pip install "qdrant-client<…"`) or migrate the repository code to `query_points`.

### 4. Start Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Check it: `curl http://localhost:6333/healthz`. The collection (`datasets`) and its payload indexes are created automatically on first use.

### 5. Install Ollama and pull the embedding model

```bash
curl -fsSL https://ollama.com/install.sh | sh     # installs and starts the service
ollama pull nomic-embed-text                       # embeddings (required)
```

Only if you want a local LLM:

```bash
ollama pull llama3.1:8b       # default for the agent
ollama pull llama3            # default for the one-shot /query/natural endpoint
```

Check it: `curl http://localhost:11434/api/tags`.

> Use the **same embedding model for ingestion and for querying**. Vectors from different models are not comparable. If you change the model, re-ingest every dataset.

---

## Configuration

The two parts read configuration **differently**, and several variable names are shared. Read this section once before you set anything.

### Catalog (Flask): `app/config.py`

`run.py` does **not** load a `.env` file. `Config` reads plain environment variables, so export them in your shell (or use `set -a; source .env; set +a`).

| Variable | Default | Used for |
|---|---|---|
| `QDRANT_HOST`, `QDRANT_PORT` | `localhost`, `6333` | Qdrant connection |
| `QDRANT_COLLECTION` | `datasets` | Collection name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server (embeddings) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (**required for ingestion**) |
| `FLASK_HOST`, `FLASK_PORT`, `FLASK_DEBUG` | `0.0.0.0`, `5000`, `false` | Web server |
| `NLQ_SCORE_THRESHOLD` | `0.6` | Minimum semantic score (also read by the agent's catalog adapter) |
| `LLM_BACKEND` | `ollama` | **One-shot NL query only.** `ollama` or `anthropic` |
| `OLLAMA_MODEL` | `llama3` | **One-shot NL query only** |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | **One-shot NL query only** (needs `ANTHROPIC_API_KEY`) |
| `NLQ_BBOX_PADDING`, `NLQ_LIMIT` | `0.5`, `10` | **One-shot NL query only** |

The LLM settings (`LLM_BACKEND`, `OLLAMA_MODEL`, `ANTHROPIC_MODEL`, `NLQ_BBOX_PADDING`, `NLQ_LIMIT`) matter only for the older one-shot natural-language endpoint `POST /query/natural`. If you only ingest, browse, use structured search, or use the agent, you can ignore them.

### Agent: environment and `.env`

`run_agent.py` loads a `.env` file from the working directory **when it creates the LLM**. Create one:

```bash
cat > .env <<'EOF'
# --- LLM backend for the agent: ollama or groq ---
LLM_BACKEND=ollama

# Ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_NUM_CTX=16384
OLLAMA_HOST=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m

# Groq
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
# GROQ_REASONING_EFFORT=medium

# Remote sources: retry once when opening fails with a transient error
OPEN_RETRIES=1
OPEN_RETRY_DELAY=10
EOF
```

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `groq` (the `--backend` flag overrides it) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model for the agent |
| `OLLAMA_NUM_CTX` | `16384` | Ollama context window in tokens. Ollama's own default is much smaller and silently truncates tool schemas, so keep this high |
| `OLLAMA_HOST` | Ollama default | Base URL for the agent's chat model |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded |
| `GROQ_API_KEY` | none | Required for `--backend groq` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model id (set by variable, because Groq retires models) |
| `GROQ_REASONING_EFFORT` | unset | Optional, passed to Groq reasoning models |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model used for catalog search. Must match ingestion |
| `QDRANT_HOST`, `QDRANT_PORT` | `localhost`, `6333` | Qdrant connection |
| `NLQ_SCORE_THRESHOLD` | `0.6` | Minimum semantic score in `search_catalog` |
| `OPEN_RETRIES`, `OPEN_RETRY_DELAY` | `1`, `10` | Retry count and delay (seconds) when opening a remote source |

### Things that can trip you up

- **`LLM_BACKEND` means different things in the two parts.** The Flask app accepts `ollama` or `anthropic`. The agent accepts `ollama` or `groq`. If you export `LLM_BACKEND=groq` in the shell that runs `run.py`, the one-shot endpoint will fail. Prefer the `--backend` flag for the agent, or keep `LLM_BACKEND` out of the shell that runs Flask.
- **`OLLAMA_MODEL` is shared** but has different defaults (`llama3` for Flask, `llama3.1:8b` for the agent). If you set it, both parts use it.
- **Qdrant and embedding settings in `.env` are not seen by the agent's catalog.** The catalog is built before `.env` is loaded, so `QDRANT_HOST`, `QDRANT_PORT` and `EMBED_MODEL` must be **exported in the shell** if you change them. The defaults work for a local setup.
- **The agent always uses the `datasets` collection** and an embedder at `http://localhost:11434`. Changing `QDRANT_COLLECTION` or the Ollama host for Flask does not affect the agent.
- **Embedding model names differ:** Flask uses `OLLAMA_EMBED_MODEL`, the agent uses `EMBED_MODEL`. Keep both the same.

---

## Part 1: run the catalog (`run.py`)

This is the original GeoMeta: ingestion, browsing, structured search and the one-shot natural-language search.

### Start it

Make sure Qdrant and Ollama are running, then:

```bash
source .venv/bin/activate
python run.py
```

Open <http://localhost:5000>.

`run.py` binds to `0.0.0.0:5000` by default (`FLASK_HOST`, `FLASK_PORT`). That makes it reachable from other machines. For local use, run it with `FLASK_HOST=127.0.0.1 python run.py`.

For something sturdier than Flask's development server, `run.py` exposes `app`, so you can use a WSGI server:

```bash
pip install gunicorn
gunicorn -w 2 -b 127.0.0.1:5000 run:app
```

### Add datasets (ingestion)

The catalog is empty until you ingest datasets. In the web interface, use **Extract** and submit a local file path or a URL. The same operation is available as `POST /datasets/extract` (see `app/routes/datasets.py` for the request body).

Supported sources:

| Format | Sources |
|---|---|
| NetCDF | local files, OPeNDAP (`http(s)://…`), remote files via fsspec |
| Zarr | local directories, `http(s)`, `s3`, `gcs` (consolidated or not) |

For each dataset, ingestion stores the title and description, spatial and temporal extent, dimensions, global attributes, and for every variable its units, names and statistics (min, max, mean, std, NaN count). It also creates the embeddings. Ingestion of large remote datasets can take a while.

### Search and browse

- **Search tab:** natural language (the LLM-extracted filters are shown) or structured filters.
- **Browse tab:** the full catalog, a format filter, a refresh button. Clicking a card opens the detail panel with the extent, a variable table with statistics, dimensions and global attributes, and a delete button.

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/datasets/extract` | Extract metadata, statistics and embeddings from a file or URL |
| `GET` | `/datasets/` | List all datasets (optional `?format=`) |
| `GET` | `/datasets/<id>` | One dataset with variable statistics |
| `DELETE` | `/datasets/<id>` | Remove the dataset and all its variable points |
| `POST` | `/query/structured` | Filter-only query (format, bounding box, time range, variables) |
| `POST` | `/query/natural` | One-shot natural-language query (needs the LLM settings above) |
| `GET` | `/health` | Health check |
| `GET` | `/` | The web interface |

```bash
curl http://localhost:5000/health
curl "http://localhost:5000/datasets/?format=zarr"
```

### The one-shot natural-language query

`POST /query/natural` sends the question to an LLM once, which returns structured filters (format, region, time expression). Code resolves the time expression into dates, the raw question is embedded for topical matching, and the filters are applied as Qdrant pre-filters. It does one LLM call and has no tool loop. The agent in Part 2 replaces it for anything beyond a simple lookup.

---

## Part 2: run the agent (`run_agent.py`)

### Prerequisites

1. Qdrant is running and **the catalog contains datasets** (Part 1).
2. Ollama is running with the embedding model pulled (the agent embeds your query).
3. An LLM backend is available: either Groq (`GROQ_API_KEY`) or a tool-calling model in Ollama.

### Run it

```bash
source .venv/bin/activate

# Groq (best tool-calling reliability so far)
python run_agent.py "Sea surface temperature around Naples in 2026" \
    --backend groq --max-mb 500

# Local model through Ollama
python run_agent.py "Sea surface temperature for the Mediterranean in 2020" \
    --backend ollama --model llama3.1:8b --num-ctx 16384 --max-mb 200
```

The step-by-step trace goes to **stderr**, the final answer to **stdout**, so you can keep just the answer with `2>/dev/null` or save it with `> answer.txt`.

### Options

| Option | Default | Meaning |
|---|---|---|
| `task` (positional) | required | What you need, in natural language |
| `--backend {ollama,groq}` | `LLM_BACKEND`, else `ollama` | LLM backend |
| `--model` | `OLLAMA_MODEL` / `GROQ_MODEL` | Override the model id |
| `--num-ctx` | `16384` | Ollama context window in tokens (ignored for Groq) |
| `--max-steps` | `15` | Step cap; after it the agent must write a plain-text summary |
| `--max-mb` | `500` | Subset size limit in MB. A request above it is refused before anything is written |
| `--out` | `subsets` | Folder for extracted subsets |
| `--runs-dir` | `runs` | Folder for run records |
| `--max-result-chars` | `6000` | Cap on each tool result shown to the model. Lower it for small contexts |
| `--quiet` | off | Do not print the step trace |

### What the agent does

For a task like the examples above, a typical run is:

1. `resolve_time_range`: turns "2026" or "last 5 years" into exact dates (computed in code, not by the LLM).
2. `search_catalog`: topic as free text, region/period/format as structured filters.
3. `compare_datasets`: compares 2 to 4 candidates (coverage of region and period, variable match, gridded or not) and recommends one. Optional.
4. `get_dataset`: details of one dataset (extent, variables, units, chunking, statistics).
5. `estimate_subset_size`: size of the result and bytes that must be read, **without reading data**.
6. `subset_dataset`: writes the subset as Zarr or NetCDF. It is refused unless that exact request was estimated first.
7. `verify_subset`: checks the written file (empty or all-NaN data, NaN fraction, infinite or constant values, units, values outside the catalog range, coordinates outside the requested region and period).

Rules the code enforces regardless of the model:

- Step cap, then a forced plain-text wrap-up.
- Malformed or text-form tool calls are sent back (at most twice).
- A first answer with no tool use is challenged (no answering from memory). A clarifying question is allowed.
- Identical repeated calls get a note that the result will not change.
- The agent cannot finish with an unverified subset (it is nudged twice).

### Output

| Where | What |
|---|---|
| stdout | The agent's answer, then a list of **tool-reported facts** (subset id, verified flag, shape, size, path, extent) that are not written by the model, so contradictions in the answer are visible |
| `subsets/` | The extracted subset (`<subset_id>.zarr` or `.nc`) and a sidecar `<subset_id>.provenance.json` |
| `subsets/provenance.jsonl` | Log of every tool call: time, arguments, outcome, duration |
| `runs/<timestamp>.json` | The whole run: task, backend, model, answer, tool calls, token usage, timing, full message trace |

The subset id is a hash of dataset, request and output format, so an identical request gives the same id. The source and the request are also written into the output file's global attributes (`geometa_subset_source`, `geometa_subset_request`).

The last line on stderr summarises the run, for example `[final_answer / delivered] 9 steps, 8 tool calls, 41.2s …`. The process exits with `0` when the run succeeded and `1` otherwise.

`outcome` is one of:

| Outcome | Meaning |
|---|---|
| `delivered` | A subset was written and verified |
| `unverified` | A subset was written but never verified |
| `source_unavailable` | The remote data server failed to open after the retry (not the agent's fault) |
| `no_subset` | The run ended without producing a subset |

### Remote sources

For remote sources the agent opens the source once per run and reuses it for the estimate and the extraction. If opening fails with a transient error (HTTP 429/500/502/503/504, timeout, connection reset) it retries once after `OPEN_RETRY_DELAY` seconds. A persistent failure is reported as `source_unavailable`. Local files are never retried.

### Tips

- **Small local models:** raise `--num-ctx` if tool calls look cut off, and lower `--max-result-chars` if the context is tight. Small models often write tool calls as plain text; the loop sends those back, but results are noticeably better with a larger model.
- **Rate limits (Groq):** waits on retries are counted in the summary line (`waiting on retries`).
- **Reproducibility:** re-run the same task and compare `runs/*.json`; the tool-reported facts and provenance files show what was actually extracted.

---

## Tests

```bash
pytest -q
```

| File | Covers |
|---|---|
| `test_subsetting.py` | Subsetting and verification on synthetic data (descending latitude, 0–360 longitudes, antimeridian, non-standard calendars, stride, NetCDF and Zarr output, size estimates) |
| `test_agent_tools.py` | Tools, argument validation and coercion, provenance, the full search → estimate → subset → verify pipeline |
| `test_qdrant_catalog.py` | `QdrantCatalog` through the real repository and query engine, using in-memory Qdrant (skipped if `qdrant-client` is missing) |
| `test_agent.py` | The agent loop with a scripted fake model, and the real `ChatOllama` client against a fake Ollama HTTP server |
| `test_resilience.py` | Retry on open and the `source_unavailable` outcome |
| `test_shared_handle.py` | Shared remote handles within a run |

The tests need neither a running Qdrant server nor a real LLM. They do not cover a real Qdrant server, real Ollama or Groq models, remote Zarr or OPeNDAP subsetting, or very large datasets. One test may be reported as an expected failure (`xfail`): in-memory Qdrant silently excludes nanosecond timestamps from datetime filters.

---

## Project layout

```
.
├── run.py                      Flask entry point
├── run_agent.py                Agent CLI entry point
├── app/                        Flask application
│   ├── __init__.py             app factory (Config, embedder, repository, blueprints)
│   ├── config.py               configuration (see above)
│   ├── routes/
│   │   ├── datasets.py         /datasets/*
│   │   ├── query.py            /query/*
│   │   └── frontend.py         /
│   ├── services/
│   │   ├── extractor.py
│   │   └── nlq_qdrant.py       builds the LLM backend for the one-shot NL query
│   ├── static/{css,js}/        web interface assets
│   └── templates/index.html    single-page application
│
├── extractors.py               NetCDF and Zarr metadata extraction, source openers
├── metadata_dataclasses.py     DatasetMetadata, VariableMetadata, statistics, extents
├── embeddings.py               Embedder interface, OllamaEmbedder
├── qdrant_repository.py        Qdrant storage (dataset point + one point per variable)
├── qdrant_query_engine.py      structured and two-stage semantic queries
├── nl_query_engine_qdrant.py   one-shot natural-language engine, LLM backends
│
├── agent.py                    agent loop, system prompt, Ollama and Groq factories
├── agent_tools.py              tool framework, ToolContext, the seven tools
├── catalog_adapters.py         CatalogAdapter, InMemoryCatalog, QdrantCatalog
├── subsetting.py               plan, estimate and write subsets
├── verification.py             sanity checks on a written subset
│
├── test_*.py                   test suites
└── geometa_uml.puml            class diagram
```

---

## Known limitations

- **Only pre-ingested datasets.** The agent does not ingest sources by itself; ingest through the web app first.
- **Statistics for large variables.** The size guard in `extractors.py` is currently inactive, so every numeric variable is loaded into memory during ingestion. Large datasets may be slow or fail at ingestion.
- **Fill values in statistics.** If a dataset's fill value is not masked (for example a sentinel of −1000 in sea-ice cells), its catalog min, mean and std are wrong, and the range check in `verify_subset` cannot catch leaked fill values. `compare_datasets` flags this case.
- **Projected grids.** `x`/`y` are accepted as lat/lon candidates at ingestion, which gives a meaningless extent on projected grids; subsetting rejects them.
- **Datasets crossing ±180°** may get inverted longitude bounds in the catalog.
- **Write-time rate limits.** Retries cover opening a source only. If a server rate-limits chunk reads while writing, the error is `write_failed` and is not treated as an outage.
- **Datetime filters and nanoseconds.** Timestamps stored with nanosecond precision are excluded by in-memory Qdrant datetime filters. Verify against your server or truncate timestamps to seconds at ingestion.