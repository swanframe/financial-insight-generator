# Financial Insight Generator (FIG)

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Interface](https://img.shields.io/badge/interface-CLI-green.svg)
![LLM](https://img.shields.io/badge/LLM-optional-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-integrated-purple.svg)

Financial Insight Generator (FIG) is a modular Python toolkit that turns raw
transaction-level data (CSV/Excel) into:

- Clean, validated datasets  
- Financial KPIs and segment breakdowns  
- Human-readable insight reports  
- An interactive CLI “assistant”, optionally powered by an LLM  
- An optional vector index + RAG layer over your transactions  
- A configurable **orchestration engine** (`native` vs `langchain`) for LLM + RAG flows

Think of it as a small, extensible **junior financial analyst** you can run
locally on a modest laptop (e.g. macOS Big Sur, 4 GB RAM), with:

- A **template-only** deterministic mode (no API keys, fully offline)  
- An **LLM-only** mode for narrative reports and chat  
- A full **LLM + RAG** mode where the LLM is grounded in actual transactions via a vector store  
- The option to run LLM features through either a **native engine** or a **LangChain-based engine**

---

## Current Capabilities at a Glance

- End-to-end **data → analytics → insights** pipeline
- Deterministic **template reports** in English and Bahasa Indonesia
- Optional **LLM narrative reports** (`template`, `llm`, `hybrid` modes)
- Optional **RAG** for both reports and chat using a transaction vector index
- Config-driven architecture (`config.yaml`) with clean separation of concerns
- Fully working on a small, offline machine (dummy embeddings, no API key)
- When API keys are provided:
  - Real LLM calls (via `fig.llm_client` or the LangChain engine)
  - Real embeddings + vector search (e.g. via `chromadb`)
- **Orchestration engine switch**: `native` vs `langchain`, controllable via config or CLI
- ~45+ automated tests covering config, analytics, LLM wiring, vector store, and RAG integration

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
  - [Native vs LangChain Engines](#native-vs-langchain-engines)
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
  - [Orchestration / Engine](#orchestration--engine)
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
  - The LLM path can run via:
    - A **native engine** (`fig.llm_client` + custom RAG wiring), or
    - A **LangChain engine** (`langchain_chains.generate_report_with_langchain`).

- **Interactive assistant (CLI)**
  - Rule-based commands for:
    - Summary / overview
    - Top categories / products / customers / channels
    - Trend and anomaly
    - Time series
  - **LLM-powered free-form questions** about the data when enabled.
  - Free-form questions can optionally use **RAG** (similar transactions) as extra context.
  - Free-form questions can be answered by:
    - Native chat path (`fig.llm_chatbot`), or
    - LangChain chat chain (`langchain_chains.answer_question_with_langchain`).

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
- **LLM integration (optional):**
  - Provider-agnostic client (`fig.llm_client`, e.g. OpenAI via `openai`)
  - LangChain wrappers (`langchain-core`, `langchain-openai`) for orchestration engine `langchain`
- **Embeddings & vector DB (optional):**
  - Provider-agnostic embeddings (`openai` or a local `dummy` provider)
  - Local vector store (e.g. `chromadb`) or in-memory implementation
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
   - LLM-based via:
     - **Native engine:** `fig.llm_insights.generate_llm_report`
     - **LangChain engine:** `fig.langchain_chains.generate_report_with_langchain`
   - Both use:
     - Structured context from `metrics_bundle`
     - Optional **RAG context** (representative transactions).
5. **LLM client & prompts**
   - `fig.llm_prompts` – builds language-aware system/user prompts.
   - `fig.llm_client` – thin provider-agnostic wrapper around LLM APIs (used by native engine).
   - `fig.langchain_llm` – LangChain Chat model + embeddings wrappers (used by LangChain engine).
6. **Embeddings & vector store**
   - `fig.embeddings.embed_texts(texts, config)`:
     - `provider: "openai"` – real embeddings via API.
     - `provider: "dummy"` – deterministic local embeddings (offline).
   - `fig.vector_store`:
     - Builds and queries the vector index.
7. **RAG layer**
   - `fig.retrieval` – high-level retrieval API:
     - Builds transaction index from a DataFrame.
     - Retrieves relevant transactions given a query.
   - `fig.retrieval_schema` – `RetrievedTransaction` and `RetrievalContext` for clean, prompt-friendly RAG objects.
   - **LangChain adapter**: `fig.langchain_retriever.TransactionsRetriever` exposes the same retrieval as a LangChain `BaseRetriever` returning `Document`s.
8. **CLI & interactive assistant**
   - `src.fig.cli` – entry point for:
     - Running the pipeline.
     - Generating reports.
     - Entering interactive chat mode.
   - `fig.chatbot` – high-level chat loop + command routing.
   - **Free-form LLM logic**:
     - Native: `fig.llm_chatbot.answer_freeform_question`.
     - LangChain: `fig.langchain_chains.answer_question_with_langchain`.

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
  (RetrievedTransaction / RetrievalContext)
           ├────────── llm_insights.generate_llm_report (native LLM + RAG)
           │
           └────────── langchain_chains.generate_report_with_langchain (LangChain RAG chain)
                                     ↓
                             CLI / chat (src.fig.cli, fig.chatbot)
                                     │
                                     ├─ llm_chatbot.answer_freeform_question (native)
                                     └─ langchain_chains.answer_question_with_langchain (LangChain)
````

### Native vs LangChain Engines

Two orthogonal switches control behaviour:

1. **LLM mode** (`config.llm.mode`):

   * `template` – only template-based report; no LLM calls.
   * `llm` – LLM-only narrative report (optionally RAG-grounded).
   * `hybrid` – template report + LLM refinement (optionally RAG-grounded).

2. **Orchestration engine** (`config.orchestration.engine` or CLI `--engine`):

   * `native` – use the original Python pipeline:

     * `llm_insights.generate_llm_report`
     * `llm_chatbot.answer_freeform_question`
   * `langchain` – use LangChain-based chains:

     * `langchain_chains.generate_report_with_langchain`
     * `langchain_chains.answer_question_with_langchain`

This lets you demonstrate both “classic” RAG wiring and modern LangChain patterns in a single project.

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
│       ├── langchain_llm.py
│       ├── langchain_documents.py
│       ├── langchain_retriever.py
│       ├── langchain_chains.py
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

   * Set `PYTHONPATH`:

     ```bash
     export PYTHONPATH=src
     ```

   * Or prefix commands with `PYTHONPATH=src`.

---

## Configuration

All configuration lives in `config.yaml` and is loaded via `fig.config.load_config`.

### Data

```yaml
data:
  input_path: "data/raw/sample_transactions.csv"
  date_format: "%Y-%m-%d"
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

Adjust these to match your raw file.

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

This controls:

* Template-based reports
* CLI banners and messages
* Chat prompts
* LLM instructions on which language to answer in

### LLM

The LLM section is **optional**. If omitted, FIG runs in pure template mode with `llm.enabled = false`.

```yaml
llm:
  enabled: false          # master toggle; false = no LLM calls

  provider: "openai"      # logical provider name ("openai", "dummy", etc.)
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

```yaml
embeddings:
  provider: "openai"              # "openai" or "dummy"
  model: "text-embedding-3-small" # ignored for dummy, used for real providers
  api_key_env_var: "OPENAI_API_KEY"
  timeout_seconds: 30
```

`provider: "dummy"` uses a fully local, deterministic embedding (no network).

### Vector Store

```yaml
vector_store:
  enabled: false                  # master toggle for vector features

  provider: "chroma"              # "chroma" (persistent) or "in_memory"
  persist_path: "data/vector_store"
  collection_name: "fig_transactions"

  default_top_k: 5
```

### Orchestration / Engine

This is where you choose between the **native** and **LangChain** orchestration:

```yaml
orchestration:
  engine: "native"    # "native" or "langchain"
```

* `native` – use `llm_insights` + `llm_chatbot` directly.
* `langchain` – use `langchain_chains` + `langchain_retriever` and LangChain’s `Runnable` pipelines.

You can override this via CLI `--engine`.

---

## Usage

> Assume `PYTHONPATH=src` is set in the examples below.

### One-off report script

Run the full pipeline + report generation:

```bash
python run_insights_report.py --config config.yaml
```

Options:

```bash
python run_insights_report.py \
  --config config.yaml \
  --lang en \
  --report-mode template    # or llm / hybrid \
  --engine native           # or langchain
```

* `--engine` only matters when `llm.enabled: true` and `mode` is `llm` or `hybrid`.

### CLI

Main CLI entrypoint:

```bash
python -m src.fig.cli --config config.yaml
```

Common flags:

* `--no-report` – skip printing the report (useful for chat only).
* `--interactive` – start interactive chat after report.
* `--lang` – override language.
* `--report-mode {template,llm,hybrid}` – override LLM mode.
* `--engine {native,langchain}` – choose orchestration engine.

Example:

```bash
python -m src.fig.cli --config config.yaml --engine langchain --interactive
```

### LLM report modes

Effective report mode:

> CLI `--report-mode` → `llm.mode` in config → `"template"`

* `template`
  Uses `insights.generate_full_report`. No LLM calls.

* `llm`

  * Native engine:

    * `llm_insights.generate_llm_report(..., mode_override="llm")`
  * LangChain engine:

    * `langchain_chains.generate_report_with_langchain(..., template_report=None)`

* `hybrid`

  * Native engine:

    * template report first, then `generate_llm_report(..., mode_override="hybrid")`
  * LangChain engine:

    * template report first, then `generate_report_with_langchain(..., template_report=template_report)`

If `llm.enabled: false`, LLM modes fall back to template with a note.

### Interactive assistant (chat)

Start interactive mode:

```bash
python -m src.fig.cli --config config.yaml --interactive
```

#### Rule-based commands

Handled by `fig.chatbot` and/or `fig.llm_chatbot` (non-LLM logic):

* `/summary`
* `/trend`
* `/help`
* `/quit` or `/exit`

(Plus other commands depending on chat implementation: overview, top segments, etc.)

These work the same regardless of engine.

#### Free-form questions

Anything else is treated as a free-form question:

* Native engine:

  * `fig.llm_chatbot.answer_freeform_question(...)`
* LangChain engine:

  * `langchain_chains.answer_question_with_langchain(...)`

Both can optionally include RAG context if `vector_store.enabled: true` and an index exists.

### Multilingual usage (EN / ID)

Use `ui.language` in config:

```yaml
ui:
  language: "id"
```

Or override via CLI:

```bash
python run_insights_report.py --config config.yaml --lang id
python -m src.fig.cli --config config.yaml --lang id
```

The language is respected for:

* Template reports
* CLI text
* LLM prompts (native + LangChain)

### Vector index build & search

Enable in config:

```yaml
vector_store:
  enabled: true
  provider: "chroma"
  persist_path: "data/vector_store"
  collection_name: "fig_transactions"
```

Choose embeddings:

```yaml
embeddings:
  provider: "openai"   # or "dummy" for local-only demos
```

Build index:

```bash
python run_build_vector_index.py --config config.yaml
```

Search:

```bash
python run_vector_search.py \
  --config config.yaml \
  --query "large electronics orders in March"
```

The same index is reused by RAG in both native and LangChain engines.

### RAG debug & observability (FIG_DEBUG_RAG)

Turn on verbose RAG logging:

```bash
export FIG_DEBUG_RAG=1
```

You’ll see RAG information printed whenever:

* The index is built.
* RAG is used for reports.
* RAG is used for free-form chat questions.

Disable:

```bash
unset FIG_DEBUG_RAG
```

---

## Quickstart

### Mode A – Template-only (fully offline)

```yaml
llm:
  enabled: false

embeddings:
  provider: "dummy"

vector_store:
  enabled: false
```

Run:

```bash
python run_insights_report.py --config config.yaml
python -m src.fig.cli --config config.yaml --interactive
```

### Mode B – LLM only

```yaml
llm:
  enabled: true
  provider: "openai"
  mode: "llm"

vector_store:
  enabled: false

orchestration:
  engine: "native"    # or "langchain"
```

Export API key and run report + CLI.

### Mode C – LLM + RAG

```yaml
llm:
  enabled: true
  provider: "openai"
  mode: "llm"

embeddings:
  provider: "openai"

vector_store:
  enabled: true
  provider: "chroma"

orchestration:
  engine: "native"    # or "langchain"
```

Then:

```bash
python run_build_vector_index.py --config config.yaml
python run_insights_report.py --config config.yaml --report-mode llm
python -m src.fig.cli --config config.yaml --interactive
```

---

## Offline vs API-backed features

**Fully offline / local:**

* Data pipeline, analytics, and template reports.
* CLI rule-based commands (`/summary`, `/trend`, etc.).
* `embeddings.provider: "dummy"` (local deterministic vectors).
* `vector_store.provider: "in_memory"` (in-process store).
* Vector index build + similarity search using dummy embeddings.
* LangChain engine also works offline with `provider: "dummy"`.

**Requires an API key:**

* `llm.enabled: true` with `provider: "openai"` (or another cloud LLM).
* `embeddings.provider: "openai"` for real embeddings.
* Native and LangChain engines both use the same `Config.llm` and `Config.embeddings`.

---

## Testing

Run the full test suite:

```bash
pytest
```

You should see all tests pass (around ~47 tests), with only minor pandas
FutureWarnings about `"M"` vs `"ME"` resampling.

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
* Integration tests that assert RAG context is actually injected into:

  * LLM report prompts.
  * Chat prompts.

---

## AI Engineering Highlights

* **Structured analytics first, LLM second**

  * Clear separation between data/analytics and LLM layers.
  * LLM (native or LangChain) operates on a rich `metrics_bundle` and optional RAG context.

* **Config-driven architecture**

  * Behaviour controlled via `config.yaml`.
  * Easy switches between template-only, LLM-only, and LLM+RAG.
  * Extra switch for orchestration engine (`native` vs `langchain`).

* **Provider-agnostic LLM client + LangChain**

  * `fig.llm_client` abstracts the LLM backend for the native path.
  * `fig.langchain_llm` wraps the same config for LangChain.
  * LLM can be disabled, misconfigured, or swapped without breaking the core system.

* **RAG-ready by design**

  * Transaction-level embeddings and vector store with a clean API.
  * `fig.retrieval` + `fig.retrieval_schema` provide a stable RAG contract.
  * Native and LangChain report/chat both reuse the same RAG layer.

* **Multilingual UX**

  * English / Bahasa Indonesia support for reports, chat, and prompts.
  * LLM is explicitly instructed which language to use.

* **Observability & safety**

  * `FIG_DEBUG_RAG` for verbose RAG introspection.
  * Graceful fallbacks when LLM or vector store configuration is missing or invalid.
  * Clear, user-friendly notes when LLM features are disabled.

---

## Future Enhancements

Some potential extensions on top of the existing architecture:

* **Docs & diagrams**

  * Add a `docs/` folder with architecture diagrams and design notes.

* **“Show evidence” in chat**

  * Add a command to display the last RAG block (e.g. `/evidence`).

* **Web API wrapper**

  * Wrap FIG in a small FastAPI or Flask service exposing metrics, reports, chat, and search.

* **Simple UI**

  * Attach a small dashboard (Streamlit, React) on top of the existing CLI + RAG backend.

---

## License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

## Author

**Rahman**

* GitHub: [https://github.com/swanframe](https://github.com/swanframe)
* LinkedIn: [https://www.linkedin.com/in/rahman-080902337](https://www.linkedin.com/in/rahman-080902337)