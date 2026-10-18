# 🌍 GeoMeta

**GeoMeta** is a geospatial scientific dataset catalog for **NetCDF** and **Zarr**.

It extracts metadata and variable statistics, indexes datasets spatially and temporally, and uses semantic search + an LLM to make scientific datasets easy to find.

## ✨ Features

* 📦 **NetCDF & Zarr** — local and remote/OPeNDAP sources
* 📊 **Variable statistics** — mean, std, min, max, median, and NaN count
* 🔎 **Semantic search** — dataset- and variable-level embeddings
* 🗺️ **Spatial & temporal filtering** — bounding boxes and date ranges
* 🤖 **Natural-language queries** — ask for datasets in plain English
* 🧠 **Qdrant** — vector storage and similarity search
* 🌐 **Flask API + web UI** — Search, Extract, and Browse

### How search works

Each dataset gets:

* **1 dataset embedding** from its title, description, and some other global attributes
* **1 embedding per variable** from its name and CF metadata

Natural-language queries are handled in two parts. An LLM extracts structured filters (format, region, time range) from the question. Region names become a bounding box, and expressions such as “in 2026”, “last 5 years” or “since 2015” are resolved to concrete dates in code. The embedding model then matches the topic against the dataset and variable embeddings, and the filters are applied as Qdrant pre-filters. Format, bounding box and dates are never embedded, only filtered.

## 🏗️ Architecture

```text
NetCDF / Zarr
     │
     ▼
Extraction Pipeline
     │
     ▼
DatasetMetadata + VariableStatistics
     │
     ▼
Qdrant
 ┌───┴─────────────┐
 │                 │
Dataset vectors   Variable vectors
 │                 │
 └───────┬─────────┘
         ▼
     Flask API
         │
         ▼
      Web UI
```

## 🚀 Getting Started

### Requirements

* Python 3.10+
* Docker
* [Ollama](https://ollama.com/)
* Qdrant

### 1. Clone & install

```bash
git clone https://github.com/<username>/geometa.git
cd geometa

python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run --name qdrant -d \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

### 3. Set up Ollama

Install Ollama, then:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 4. Run GeoMeta

```bash
python run.py
```

Open **http://localhost:5000**.

## ⚙️ Configuration

Everything is configurable through environment variables:

| Variable              | Default                  | Purpose                 |
| --------------------- | ------------------------ | ----------------------- |
| `QDRANT_HOST`         | `localhost`              | Qdrant host             |
| `QDRANT_PORT`         | `6333`                   | Qdrant port             |
| `QDRANT_COLLECTION`   | `datasets`               | Collection name         |
| `OLLAMA_HOST`         | `http://localhost:11434` | Ollama URL              |
| `OLLAMA_MODEL`        | `llama3.1`               | LLM model               |
| `OLLAMA_EMBED_MODEL`  | `nomic-embed-text`       | Embedding model         |
| `LLM_BACKEND`         | `ollama`                 | `ollama`                |
| `NLQ_BBOX_PADDING`    | `0.5`                    | BBox padding in degrees |
| `NLQ_LIMIT`           | `10`                     | Max NLQ results         |
| `NLQ_SCORE_THRESHOLD` | `0.6`                    | Similarity threshold    |
| `DEBUG`               | `false`                  | Flask debug mode        |
| `HOST`                | `0.0.0.0`                | Flask host              |
| `PORT`                | `5000`                   | Flask port              |


## 🔌 API

### Datasets

| Method   | Endpoint            | Description               |
| -------- | ------------------- | ------------------------- |
| `POST`   | `/datasets/extract` | Extract & index a dataset |
| `GET`    | `/datasets/`        | List datasets             |
| `GET`    | `/datasets/<id>`    | Get a dataset             |
| `DELETE` | `/datasets/<id>`    | Delete a dataset          |

```bash
curl -X POST http://localhost:5000/datasets/extract \
  -H "Content-Type: application/json" \
  -d '{"source": "/data/myfile.nc", "format": "netcdf"}'
```

### Queries

```text
POST /query/structured
POST /query/natural
```

Example:

```bash
curl -X POST http://localhost:5000/query/natural \
  -H "Content-Type: application/json" \
  -d '{"query": "sea surface temperature over the Mediterranean in 2020"}'
```

Health check:

```bash
curl http://localhost:5000/health
```

## 📁 Supported Formats

| Format                  | Support                         |
| ----------------------- | ------------------------------- |
| NetCDF4 / HDF5          | `h5netcdf`                      |
| NetCDF3                 | `netcdf4` / `scipy`             |
| Remote NetCDF / OPeNDAP | `netcdf4` / `pydap`             |
| Zarr                    | consolidated & non-consolidated |
| OME-Zarr                | Zarr v3 traversal               |

## ⚠️ A couple of things to know

**Don't change the embedding model after ingestion.**
Vectors from different models cannot share the same embedding space. If you switch models, recreate the Qdrant collection and re-ingest.

**Tune `NLQ_SCORE_THRESHOLD`.**
`0.55–0.70` is a reasonable starting range for `nomic-embed-text`, but the best value depends on your catalog.

**Large variables may skip statistics.**
Statistics are computed during ingestion, but very large arrays are skipped to avoid excessive memory usage.

---

Made for scientists who would rather **ask for data than hunt for filenames.** 🌍
