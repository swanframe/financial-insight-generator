# Financial Insight Generator (FIG)

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Interface](https://img.shields.io/badge/interface-CLI-green.svg)
![LLM](https://img.shields.io/badge/LLM-optional-orange.svg)

Financial Insight Generator (FIG) is a modular Python toolkit that turns raw
transaction-level data (CSV/Excel) into:

- Clean, validated datasets  
- Financial KPIs and segment breakdowns  
- Human-readable insight reports  
- An interactive CLI “assistant”, optionally powered by an LLM  
- An optional vector index + RAG layer over your transactions

Think of it as a small, extensible **junior financial analyst** you can run
locally on a modest laptop (e.g. macOS Big Sur, 4 GB RAM), with:

- A **template-only** deterministic mode (no API keys, fully offline)  
- An **LLM-only** mode for narrative reports and chat  
- A full **LLM + RAG** mode where the LLM is grounded in actual transactions via a vector store

---

## Current Capabilities at a Glance

- End-to-end **data → analytics → insights** pipeline
- Deterministic **template reports** in English and Bahasa Indonesia
- Optional **LLM narrative reports** (`template`, `llm`, `hybrid` modes)
- Optional **RAG** for both reports and chat using a transaction vector index
- Config-driven architecture (`config.yaml`) with clean separation of concerns
- Fully working on a small, offline machine (dummy embeddings, no API key)
- When API keys are provided:
  - Real LLM calls (via `fig.llm_client`)
  - Real embeddings + vector search (via `chromadb`)
- ~45 automated tests covering config, analytics, LLM wiring, vector store, and RAG integration

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Directory Layout](#directory-layout)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Data](#data)
  - [Columns](#columns)
  - [Analytics](#analytics)
  - [Output](#output)
  - [UI & Language](#ui--language)
  - [LLM](#llm)
  - [Embeddings](#embeddings)
  - [Vector Store](#vector-store)
- [Usage](#usage)
  - [One-off report script](#one-off-report-script)
  - [CLI](#cli)
  - [LLM report modes](#llm-report-modes)
  - [Interactive assistant (chat)](#interactive-assistant-chat)
  - [Multilingual usage (EN / ID)](#multilingual-usage-en--id)
  - [Vector index build & search](#vector-index-build--search)
  - [RAG debug & observability (FIG_DEBUG_RAG)](#rag-debug--observability-fig_debug_rag)
- [Quickstart](#quickstart)
  - [Mode A – Template-only (fully offline)](#mode-a--template-only-fully-offline)
  - [Mode B – LLM only](#mode-b--llm-only)
  - [Mode C – LLM + RAG](#mode-c--llm--rag)
- [Offline vs API-backed features](#offline-vs-api-backed-features)
- [Testing](#testing)
- [AI Engineering Highlights](#ai-engineering-highlights)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Author](#author)

---

## Features

- **Data ingestion & cleaning**
  - Load CSV/Excel transaction data.
  - Validate required columns and types.
  - Normalize dates, filter invalid rows, and save a cleaned dataset.

- **Analytics & KPIs**
  - Overall metrics:
    - Revenue, cost, gross profit, gross margin.
    - Number of transactions, average order value (AOV).
    - Date range of the dataset.
  - Time series metrics (e.g., revenue by month).
  - Segment metrics by:
    - Category
    - Product
    - Customer
    - Channel
  - Trend analysis and a simple anomaly check on recent revenue.

- **Insight reports**
  - Template-based, deterministic report generation (no LLM required).
  - Optional **LLM-based narrative reports** that use analytics output as structured context.
  - Three modes: `template`, `llm`, `hybrid`.
  - In `llm` / `hybrid` modes, the LLM can be **RAG-grounded** using retrieved transactions.

- **Interactive assistant (CLI)**
  - Rule-based commands for:
    - Summary / overview
    - Top categories / products / customers / channels
    - Trend and anomaly
    - Time series
  - **LLM-powered free-form questions** about the data when enabled.
  - Free-form questions can optionally use **RAG** (similar transactions) as extra context.

- **Multilingual UX**
  - English and Bahasa Indonesia support.
  - The same language setting is respected across reports and chat.
  - LLM prompts are explicitly instructed to answer in the selected language.

- **Vector search over transactions (optional, RAG-friendly)**
  - Builds embeddings for each cleaned transaction and indexes them in a vector store.
  - Supports local-first providers (persistent Chroma or an in-memory store).
  - Exposes a CLI tool to search for transactions similar to a natural-language query.
  - The vector index is reused as **RAG context** for LLM reports and chat.

---

## Tech Stack

- **Language:** Python 3.12
- **Core libraries:** `pandas`, `numpy`
- **CLI & tooling:** standard library (`argparse`, `pathlib`)
- **Testing:** `pytest`
- **LLM integration (optional):** provider-agnostic client (`fig.llm_client`, e.g. OpenAI via `openai`)
- **Embeddings & vector DB (optional):**
  - Provider-agnostic embeddings (`openai` or a local `dummy` provider)
  - `chromadb` for persistent vector storage
- **i18n:** YAML-based localization (`src/fig/locales/en.yaml`, `src/fig/locales/id.yaml`)

---

## Architecture Overview

At a high level:

1. **Configuration** (`config.yaml` + `fig.config`)
2. **Data pipeline**
   - `fig.data_loader` – load CSV/Excel
   - Internal validation/cleaning – normalize dates, filter rows
   - Writes cleaned data to `data/processed/cleaned_transactions.csv`
3. **Analytics** (`fig.analytics`)
   - Computes KPI dictionaries and DataFrames.
   - Main output is a **`metrics_bundle`**:
     ```python
     metrics_bundle = {
         "overall": {...},          # overall KPIs
         "time_series": df,         # revenue over time
         "segments": {...},         # category / product / customer / channel
         "monthly_trend": {...},    # MoM trend summary
         "anomaly": {...},          # recent anomaly info
     }
     ```
4. **Insight generation**
   - Template-based (`fig.insights.generate_full_report`).
   - LLM-based (`fig.llm_insights.generate_llm_report`) using:
     - Structured context from `metrics_bundle`
     - Optional **RAG context** (a small set of representative transactions)
5. **LLM client & prompts**
   - `fig.llm_prompts` – builds language-aware system/user prompts.
   - `fig.llm_client` – thin provider-agnostic wrapper around LLM APIs.
6. **Embeddings & vector store**
   - `fig.embeddings.embed_texts(texts, config)`:
     - `provider: "openai"` – real embeddings via API.
     - `provider: "dummy"` – deterministic local embeddings (offline).
   - `fig.vector_store`:
     - `build_or_update_index(df, config, rebuild=False)` – index transactions.
     - `query_similar_transactions(query_text, config, top_k=None)` – nearest neighbours.
7. **RAG layer**
   - `fig.retrieval` – high-level retrieval API:
     - `build_transaction_index_from_dataframe(df, config, rebuild=False)`
     - `retrieve_transactions_for_query(query_text, config, ...)`
   - `fig.retrieval_schema` – `RetrievedTransaction` and `RetrievalContext` for clean, prompt-friendly RAG objects.
   - Used by:
     - `fig.llm_insights` for LLM reports.
     - `fig.llm_chatbot` for free-form chat questions.
8. **CLI & interactive assistant**
   - `fig.cli` – entry point for:
     - Running the pipeline.
     - Generating reports.
     - Entering interactive chat mode.
   - `fig.chatbot` – rule-based commands (summary/trend/etc.).
   - `fig.llm_chatbot` – LLM-powered free-form questions, optionally with RAG.

You can think of it as this pipeline:

```text
raw CSV/Excel
    ↓
data_loader → cleaning
    ↓
analytics.build_metrics_bundle
    ↓
           ┌────────── insights.generate_full_report (template)
           │
           │
embeddings + vector_store + retrieval  ──▶  RAG context
           │                               (RetrievedTransaction / RetrievalContext)
           └────────── llm_insights.generate_llm_report (LLM + RAG)
                                     ↓
                               CLI / chat (fig.cli, fig.chatbot, fig.llm_chatbot)
````

---

## Directory Layout

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   │   └── sample_transactions.csv
│   └── processed/
│       └── cleaned_transactions.csv      # generated by the pipeline
├── reports/
│   └── financial_insights.txt            # generated report (optional)
├── src/
│   └── fig/
│       ├── __init__.py
│       ├── analytics.py
│       ├── chatbot.py
│       ├── cli.py
│       ├── config.py
│       ├── data_loader.py
│       ├── embeddings.py
│       ├── i18n.py
│       ├── insights.py
│       ├── llm_chatbot.py
│       ├── llm_client.py
│       ├── llm_insights.py
│       ├── llm_prompts.py
│       ├── retrieval.py
│       ├── retrieval_schema.py
│       ├── vector_store.py
│       └── locales/
│           ├── en.yaml
│           └── id.yaml
├── tests/
│   ├── test_analytics.py
│   ├── test_config.py
│   ├── test_config_llm.py
│   ├── test_config_vector_store.py
│   ├── test_data_loader.py
│   ├── test_embeddings.py
│   ├── test_insights.py
│   ├── test_llm_chatbot.py
│   ├── test_llm_client.py
│   ├── test_llm_insights.py
│   ├── test_llm_prompts.py
│   ├── test_llm_rag_integration.py
│   ├── test_retrieval_api.py
│   ├── test_retrieval_schema.py
│   ├── test_vector_search_cli.py
│   └── test_vector_store.py
├── run_insights_report.py
├── run_build_vector_index.py
└── run_vector_search.py
```

---

## Installation

1. **Clone the repo** and create a virtual environment.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional, for LLM or real embeddings) Set up a provider. For OpenAI, export an API key:

   ```bash
   export OPENAI_API_KEY="sk-...your-key-here..."
   ```

4. Make sure Python can find the `src/` package. Either:

   * Set `PYTHONPATH` in your shell:

     ```bash
     export PYTHONPATH=src
     ```

     or

   * Prefix commands with `PYTHONPATH=src`, e.g.:

     ```bash
     PYTHONPATH=src python run_insights_report.py
     ```

---

## Configuration

All configuration lives in `config.yaml` and is loaded via `fig.config.load_config`.

### Data

```yaml
data:
  input_path: "data/raw/sample_transactions.csv"
  date_format: "%Y-%m-%d"  # optional; let pandas infer if omitted
  parse_dates: true
```

### Columns

Logical → physical column mapping:

```yaml
columns:
  date: "date"
  amount: "amount"
  cost: "cost"
  category: "category"
  product: "product"
  customer_id: "customer_id"
  channel: "channel"
```

If your raw file uses different column names, adjust these mappings.

### Analytics

```yaml
analytics:
  resample_freq: "M"       # monthly; see pandas offsets
  min_transactions: 10
  anomaly_window_days: 14
  anomaly_sigma_threshold: 2.0
```

### Output

```yaml
output:
  save_clean_data: true
  clean_data_path: "data/processed/cleaned_transactions.csv"

  save_metrics: true
  metrics_path: "data/processed/metrics.json"

  save_report: true
  report_path: "reports/financial_insights.txt"
```

### UI & Language

```yaml
ui:
  language: "en"   # or "id" for Bahasa Indonesia
```

This language is used for:

* Template-based reports
* CLI banners and messages
* Chatbot prompts and responses
* LLM prompts (the model is explicitly told which language to answer in)

You can also override it via CLI with `--lang`.

### LLM

The LLM section is **optional**. If omitted, FIG runs in pure template mode with `llm.enabled = false`.

Example configuration:

```yaml
llm:
  enabled: false          # master toggle; false = no LLM calls

  provider: "openai"      # logical provider name
  model: "gpt-4.1-mini"   # model name for your provider

  temperature: 0.3
  max_tokens: 800

  api_key_env_var: "OPENAI_API_KEY"
  timeout_seconds: 30

  # Report / assistant modes:
  # - "template": built-in templates only
  # - "llm":      LLM-only narrative reports (optionally RAG-grounded)
  # - "hybrid":   template report + LLM refinement (optionally RAG-grounded)
  mode: "template"

  # Max characters of structured context passed into prompts
  max_context_chars: 12000
```

> ⚠️ **Never put API keys in `config.yaml`.**
> Use environment variables instead:
>
> ```bash
> export OPENAI_API_KEY="sk-...your-key-here..."
> ```

### Embeddings

Embeddings configuration is shared by the vector store and the RAG layer:

```yaml
embeddings:
  provider: "openai"              # "openai" for real embeddings, or "dummy" for a local demo
  model: "text-embedding-3-small" # ignored for dummy, used for real providers
  api_key_env_var: "OPENAI_API_KEY"
  timeout_seconds: 30
```

Notes:

* `provider: "openai"` uses the OpenAI embeddings API:

  * `api_key_env_var` is looked up in the environment.
  * Good for realistic similarity search and RAG.
* `provider: "dummy"` uses a deterministic, local-only embedding implementation:

  * No API key and no network required.
  * Great for offline demos or when you’re out of quota.
* The same embeddings config is used by:

  * `run_build_vector_index.py`
  * `run_vector_search.py`
  * The RAG layer in `fig.llm_insights` and `fig.llm_chatbot` (when vector features are enabled).

### Vector Store

The vector store section controls whether vector features are enabled and how the index is stored:

```yaml
vector_store:
  enabled: false                  # master toggle for all vector features

  provider: "chroma"              # "chroma" (persistent) or "in_memory"
  persist_path: "data/vector_store"
  collection_name: "fig_transactions"

  default_top_k: 5
```

Notes:

* When `enabled: false`, scripts that depend on vector search (index build, vector CLI, RAG) simply skip vector logic; the rest of FIG is unaffected.
* `provider: "chroma"`:

  * Uses `chromadb` as a local-first vector database.
  * Persists the index under `persist_path`.
* `provider: "in_memory"`:

  * Uses an in-process Python implementation.
  * Does not write anything to disk (handy for tests and quick demos).
* The **RAG layer** (for reports and chat) reuses this same vector store:

  * If `vector_store.enabled: true` and the index is built, the LLM sees additional “RAG context” in its prompt.
  * If disabled or misconfigured, RAG is silently skipped and the LLM falls back to metrics-only context.

---

## Usage

> **Note:** In the examples below, assume `src` is on `PYTHONPATH` (e.g. `export PYTHONPATH=src`).

### One-off report script

Run the full pipeline + report generation:

```bash
python run_insights_report.py
```

Options:

```bash
python run_insights_report.py \
  --config config.yaml \
  --lang en \
  --report-mode template    # or llm / hybrid
```

### CLI

A convenient wrapper around the full workflow lives in `fig.cli`:

```bash
python -m fig.cli --config config.yaml
```

This will:

1. Load the config.
2. Run the data pipeline.
3. Run analytics.
4. Print the report (template or LLM, depending on config).
5. Optionally start an interactive chat session if you pass `--interactive`.

```bash
python -m fig.cli --config config.yaml --interactive
```

### LLM report modes

The effective report mode is:

> CLI `--report-mode` (if provided) → `llm.mode` in `config.yaml` → `"template"`

* `template`
  Uses `generate_full_report(metrics_bundle, language)`.
  **No LLM calls** are made, even if `llm.enabled: true`.

* `llm`
  Uses `generate_llm_report(..., mode_override="llm")`:

  * Summarizes `metrics_bundle` into compact structured context.
  * If vector features are enabled and an index exists, retrieves **representative transactions** via `fig.retrieval`.
  * Builds an LLM prompt via `fig.llm_prompts`, including both metrics and (optionally) RAG context.
  * Calls the provider via `fig.llm_client.generate_text`.

* `hybrid`
  Uses `generate_llm_report(..., mode_override="hybrid")`:

  * First builds the template report.
  * Passes both the report and metrics into the LLM.
  * Optionally includes RAG context if available.
  * The LLM refines/rewrites the report into a more narrative style.

If `llm.enabled: false`, the LLM modes automatically fall back to the template report with a short explanatory note.

### Interactive assistant (chat)

Start interactive mode:

```bash
python -m fig.cli --config config.yaml --interactive
```

#### Rule-based commands

The following commands are recognized by the rule-based chatbot:

* `summary`
* `overview`
* `top categories`
* `top products`
* `top customers`
* `top channels`
* `trend`
* `anomaly`
* `time series`
* `help`
* `exit`

These are handled by deterministic logic in `fig.chatbot` and behave the same whether or not the LLM is enabled.

#### Free-form questions (LLM + optional RAG)

If `llm.enabled: true`, any input that does **not** match a known command is treated as a free-form question and routed through `fig.llm_chatbot`, for example:

* `Is there anything unusual about recent sales?`
* `Why did revenue increase in March?`
* `Which customers contributed most to the last anomaly?`

When vector features are enabled (`vector_store.enabled: true` and the index is built):

* The question + high-level metrics are turned into a **retrieval query**.
* `fig.retrieval.retrieve_transactions_for_query(...)` returns a small set of relevant transactions.
* These are summarized into a compact `[RAG context: ...]` block and appended to the LLM prompt.
* The LLM answers with the benefit of both **structured metrics** and **concrete examples**.

If vector features are disabled or misconfigured, chat still works; it just becomes **LLM-only** (metrics-based, no RAG).

### Multilingual usage (EN / ID)

The same CLI commands work for both English and Bahasa Indonesia.

Examples:

* English report and chat:

  ```bash
  python -m fig.cli --config config.yaml --lang en
  ```

* Indonesian report and chat:

  ```bash
  python -m fig.cli --config config.yaml --lang id
  ```

For Indonesian:

* Template reports use translated headings and phrases from `locales/id.yaml`.
* Chat messages (`help`, `welcome`, `unknown_command`, etc.) are in Bahasa Indonesia.
* LLM prompts explicitly ask the model to respond in Bahasa Indonesia.

### Vector index build & search

To use vector / RAG features, you need to:

1. Enable the vector store in `config.yaml`:

   ```yaml
   vector_store:
     enabled: true
     provider: "chroma"         # or "in_memory"
     persist_path: "data/vector_store"
     collection_name: "fig_transactions"
   ```

2. Choose an embeddings provider (real or dummy):

   ```yaml
   embeddings:
     provider: "openai"         # or "dummy" for local-only demos
   ```

3. Build the index:

   ```bash
   python run_build_vector_index.py --config config.yaml
   ```

4. Run a similarity search via CLI:

   ```bash
   python run_vector_search.py \
     --config config.yaml \
     --query "large transactions in March"
   ```

This prints a ranked list of matching transactions with similarity scores and metadata.

### RAG debug & observability (`FIG_DEBUG_RAG`)

For deeper visibility into RAG behaviour, you can turn on an opt-in debug mode:

```bash
export FIG_DEBUG_RAG=1
```

With `FIG_DEBUG_RAG` set:

* Index builds print lines like:

  ```text
  [FIG RAG] Built vector index for 35 rows using provider 'chroma' (rebuild=False)
  ```

* Retrieval calls print lines like:

  ```text
  [FIG RAG] Retrieved 5 matches for query 'transactions that best explain the following aspects of the data: ...' (top3: txn-10:0.002, txn-12:0.001, txn-18:0.001)
  [FIG RAG] Report retrieval context:
  [FIG RAG] [RAG context: sample transactions]
  [FIG RAG] 1. 2024-01-15T00:00:00 | category=Electronics | product=Noise Cancelling Headphones | amount=180.00 | similarity=0.002
  ...
  ```

* For chat questions, you’ll see:

  ```text
  [FIG RAG] Chat retrieval context:
  [FIG RAG] [RAG context: relevant transactions]
  ...
  ```

Internally, the last chat RAG summary is also stored on the context as:

```python
context["last_rag_summary"]
```

which can be reused later for “show evidence” UX.

Unset the variable to disable debug output:

```bash
unset FIG_DEBUG_RAG
```

---

## Quickstart

### Mode A – Template-only (fully offline)

*No API key, no network, minimal RAM — works on macOS Big Sur with 4 GB.*

1. In `config.yaml`:

   ```yaml
   llm:
     enabled: false

   embeddings:
     provider: "dummy"

   vector_store:
     enabled: false
   ```

2. Run:

   ```bash
   python run_insights_report.py
   python -m fig.cli --config config.yaml --interactive
   ```

You’ll get:

* Cleaned data saved to `data/processed/cleaned_transactions.csv`.
* Analytics and a deterministic insight report.
* An interactive CLI with rule-based commands only (no LLM, no RAG).

### Mode B – LLM only

*LLM narrative, but no vector / RAG. Requires an API key, but no vector index.*

1. In `config.yaml`:

   ```yaml
   llm:
     enabled: true
     mode: "llm"      # or "hybrid"
   vector_store:
     enabled: false
   ```

2. Export your API key:

   ```bash
   export OPENAI_API_KEY="sk-...your-key-here..."
   ```

3. Run:

   ```bash
   python run_insights_report.py --config config.yaml
   python -m fig.cli --config config.yaml --interactive
   ```

The report and free-form chat answers will be LLM-generated based on metrics, but without RAG context.

### Mode C – LLM + RAG

*LLM narrative + grounded in real transactions. Requires an API key and local vector index.*

1. In `config.yaml`:

   ```yaml
   llm:
     enabled: true
     mode: "llm"      # or "hybrid"

   embeddings:
     provider: "openai"   # or "dummy" if you only want local RAG demos

   vector_store:
     enabled: true
     provider: "chroma"   # or "in_memory"
   ```

2. Export your API key (for real embeddings + LLM):

   ```bash
   export OPENAI_API_KEY="sk-...your-key-here..."
   ```

3. Build the index:

   ```bash
   python run_build_vector_index.py --config config.yaml
   ```

4. Run the report + chat:

   ```bash
   python run_insights_report.py --config config.yaml --report-mode llm
   python -m fig.cli --config config.yaml --interactive
   ```

Now:

* The report is LLM-generated and includes a small RAG context block in the prompt.
* Free-form chat questions use both metrics and relevant transactions.

If anything goes wrong in the vector layer, FIG quietly falls back to metrics-only LLM behaviour.

---

## Offline vs API-backed features

**Fully offline / local:**

* Data pipeline, analytics, and template reports.
* CLI rule-based commands (`summary`, `trend`, etc.).
* `embeddings.provider: "dummy"` (local deterministic vectors).
* `vector_store.provider: "in_memory"` (in-process store).
* Vector index build + similarity search using dummy embeddings.

**Requires an API key:**

* `llm.enabled: true` with `provider: "openai"` (or another cloud LLM).
* `embeddings.provider: "openai"` for real embeddings.
* RAG quality (semantic similarity) improves with real embeddings, but you can still prototype the pipeline with dummy embeddings.

This separation makes FIG usable as:

* A **completely offline analytics + CLI tool** on a small laptop.
* A **full LLM + RAG project** when you have an API key and want to showcase AI engineering skills.

---

## Testing

Run the full test suite with:

```bash
pytest
```

You should see all tests pass (around 45 tests), with only minor pandas FutureWarnings about `"M"` vs `"ME"` resampling.

The test suite covers:

* Config loading and validation.
* Data loading, cleaning, and saving.
* Analytics and the `metrics_bundle`.
* Template-based insight generation.
* LLM prompts and LLM client behaviour (including error paths).
* Chat routing (rule-based vs LLM).
* Embeddings (OpenAI vs dummy) and configuration error handling.
* Vector store configuration, in-memory implementation, and CLI.
* RAG schema (`fig.retrieval_schema`) and retrieval API (`fig.retrieval`).
* Integration tests that assert **RAG context is actually injected** into:

  * LLM report prompts.
  * Chat prompts.

From a hiring / portfolio perspective, the tests show:

* Config-driven, modular design.
* Components that work both **with** and **without** external services.
* Robust handling of misconfiguration, missing keys, and provider errors.
* Validation of **prompt wiring** and **RAG integration** without hitting external APIs.

---

## AI Engineering Highlights

* **Structured analytics first, LLM second**

  * Clear separation between data/analytics and LLM layers.
  * LLM operates on a rich `metrics_bundle` and optional RAG context.

* **Config-driven architecture**

  * Behaviour controlled via `config.yaml`.
  * Easy switches between template-only, LLM-only, and LLM+RAG.

* **Provider-agnostic LLM client**

  * `fig.llm_client` abstracts the LLM backend.
  * LLM can be disabled, misconfigured, or swapped without breaking the core system.

* **RAG-ready by design**

  * Transaction-level embeddings and vector store with a clean API.
  * `fig.retrieval` and `fig.retrieval_schema` provide a stable RAG contract.
  * LLM report & chat automatically pull in relevant transactions when RAG is enabled.

* **Multilingual UX**

  * English / Bahasa Indonesia support for reports, chat, and prompts.
  * LLM is explicitly instructed which language to use.

* **Observability & safety**

  * `FIG_DEBUG_RAG` for verbose RAG introspection.
  * Graceful fallbacks when LLM or vector store configuration is missing or invalid.
  * Clear, user-friendly notes when LLM features are disabled.

In short, this is a **complete RAG-style AI engineering project**:

* Realistic financial analytics.
* Optional LLM narrative layer.
* Vector-based retrieval over actual transactions.
* Tested, debuggable, and runnable on a modest local machine.

---

## Future Enhancements

The core system is already end-to-end and usable. Some **optional extensions** that could be added later:

* **Documentation & diagrams**

  * Add a `docs/` folder with:

    * A short architecture diagram.
    * Example prompts and answers.
    * A “design decisions” write-up.

* **“Show evidence” in chat**

  * Add a CLI command to display the last RAG block, e.g.:

    * `evidence` → prints `context["last_rag_summary"]`.

* **Web API wrapper**

  * Wrap FIG in a small FastAPI or Flask service exposing:

    * `/metrics`
    * `/report`
    * `/chat`
    * `/search`

* **Simple UI**

  * Attach a minimal React or Streamlit dashboard:

    * KPI cards + charts.
    * LLM + RAG answers.
    * Inspectable evidence list.

These are natural next steps on top of the existing architecture; they don’t require rewriting the core, just building on what’s already there.

---

## License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

## Author

**Rahman**

* GitHub: [https://github.com/swanframe](https://github.com/swanframe)
* LinkedIn: [https://www.linkedin.com/in/rahman-080902337](https://www.linkedin.com/in/rahman-080902337)