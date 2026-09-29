# Healthcare Insurance Assistant — Local RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers member, provider,
and policy questions using your official insurance plan documents (Evidence
of Coverage, Summary of Benefits, and related policy manuals). Runs on a single
machine with no cloud infrastructure: the only external service is the
Gemini API (embeddings + answers), and no model is ever downloaded.

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (Python)                            │
│                                                                           │
│  data/pdfs/*.pdf                                                         │
│        │                                                                 │
│        ▼                                                                 │
│  ingestion/pdf_loader.py        [PyMuPDF: load + parse, page-by-page]   │
│        │                                                                 │
│        ▼                                                                 │
│  chunking/chunker.py            [chapter/section-aware for structured   │
│        │                          docs (structure_chunker.py), else     │
│        │                          RecursiveCharacterTextSplitter 1000/200│
│        │                          + file/page/type/chapter/section meta]│
│        ▼                                                                 │
│  embeddings/embedding_service.py [pluggable: Gemini (gemini-embedding-  │
│        │                          001) or Databricks]                   │
│        ▼                                                                 │
│  vector_store/                                                          │
│    chroma_manager.py            [ChromaDB: similarity search index]     │
│    metadata_table.py            [SQLite: "Delta Table" analog, audit +  │
│                                    idempotent re-ingestion tracking]     │
│        │                                                                 │
│        ▼                                                                 │
│  rag_pipeline/                                                          │
│    retrieval_service.py    top-K + score threshold + hybrid (BM25)      │
│    multi_query.py           optional: paraphrase + fuse (RRF)           │
│    query_rewriter.py        follow-up question -> standalone question   │
│    multi_hop.py             optional: follow-up retrieval round         │
│    memory.py                 capped conversation history                │
│    guardrails.py            input/output safety checks                  │
│    prompt_templates.py      grounded system prompt + citation format    │
│    rag_pipeline.py           RAGPipeline: retrieve→context→generate     │
│        │                                                                 │
│        ▼                                                                 │
│  Gemini Chat Model (GEMINI_CHAT_MODEL) ──► grounded, cited answer       │
│        │                                                                 │
└────────┼──────────────────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FRONTEND: frontend/app.py (Streamlit)                │
│         Chat box only — no upload UI. Ingestion is fully backend-owned.  │
│         Shows: answer, expandable source citations, confidence, a        │
│         grounding-check warning when flagged, sidebar (model info,       │
│         indexed chunk count), clear-chat.                                 │
└─────────────────────────────────────────────────────────────────────────┘

Observability: utils/mlflow_tracking.py logs every question as a local
MLflow run (params, latency, token estimates, retrieved sources) — browse
with `mlflow ui` from the project root.
```

## 2. Why these choices (and what changed from a generic template)

This project reconciles a request that mixed two incompatible worlds: a
Databricks/Azure-OpenAI-flavored architecture spec, and an actual ask to
"run locally as easily as possible" with a Gemini API key. Local equivalents
were used everywhere a heavy managed service was implied:

| Spec asked for | This project uses | Why |
|---|---|---|
| Databricks Vector Search | **ChromaDB** (on-disk) | Zero setup, no cluster, same "index + similarity search" role |
| Databricks Delta Table | **SQLite** (`vector_store/metadata_table.py`) | Same schema, same structured/queryable role, zero server |
| Databricks Secret Scope | **`.env` file** (gitignored) | `config/settings.py` isolates all secret reads to one file, so swapping the *source* later is a one-file change |
| Azure OpenAI / `text-embedding-3-large` | **Pluggable provider**: Gemini `gemini-embedding-001` (default) or Databricks | Matches the actual Gemini-only key you have, and needs no model download — works on networks that block `huggingface.co` |
| MLflow on a Databricks tracking server | **MLflow, local file store** (`mlruns/`) | Identical `mlflow.log_*` API; only the tracking URI differs |

## 3. Setup (unrestricted machine — recommended path)

```bash
git clone <this repo>
cd rag_chatbot
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # no torch / local models -- small install

cp .env.example .env
# Edit .env and set GEMINI_API_KEY (get one free at https://aistudio.google.com/apikey)

python -m scripts.ingest           # one-time: indexes data/pdfs/*.pdf (a few
                                   # minutes on the free tier -- see section 7)
streamlit run frontend/app.py      # opens the chat UI in your browser
```

Add your own PDFs by dropping them into `data/pdfs/` and re-running
`python -m scripts.ingest` (or just restarting the Streamlit app — it
auto-detects new/changed files via a content hash and only processes what's
new).

## 4. Switching embedding providers

Everything routes through `embeddings/embedding_service.py`, so this is a
one-line change in `.env`:

```bash
EMBEDDING_PROVIDER=gemini      # hosted, uses your GEMINI_API_KEY (default)
EMBEDDING_PROVIDER=databricks  # Databricks-hosted, keyless inside a workspace
```

**Important:** switching providers (or `GEMINI_EMBEDDING_MODEL`) changes the
vector space. Delete `vectorstore_db/` (keep `feedback.jsonl` if you want
your feedback history) and re-run `python -m scripts.ingest` — old vectors
from one model are not comparable to query vectors from another.

## 5. Evaluating retrieval quality

To measure the full, deployed retrieval pipeline (the one real questions
actually go through) against hand-labeled ground truth
(`scripts/eval_questions.py`), run:

```bash
python -m scripts.evaluate_retrieval   # needs the vector store already populated
```

This uses whichever provider/settings are currently in `.env` and reports
Recall@1/3/5, Precision@1/3/5, NDCG@1/3/5, and MRR. It also saves every
individual question's result (not just the average) to
`eval_results/retrieval_eval_<timestamp>.json`, so a specific regression
can be traced back by name, and prints which questions missed the correct
page within `TOP_K`. The metric formulas live in `utils/retrieval_metrics.py`.

**Reviewing metrics in git, without CI:** every run also overwrites
`eval_results/latest.json` — the one snapshot in that folder that ISN'T
gitignored, specifically so it can be committed and diffed in a PR like any
other file. It also records a `corpus_fingerprint` (a hash of every file in
`data/pdfs/`) plus the active `embedding_model`, so anyone reviewing a
commit of this file can tell at a glance whether a change in the numbers was
caused by different documents or a different model. This is deliberately a
manual step: re-run `python -m scripts.evaluate_retrieval` and commit
`eval_results/latest.json` whenever you change `data/pdfs/` or the embedding
provider/model. Wiring an actual CI trigger for this is a reasonable next
step, but is a separate, bigger change (needs `GEMINI_API_KEY` as a repo
secret and network access in CI) that hasn't been built here.

## 6. Fully keyless mode: replacing Gemini with Databricks-hosted models

Every LLM API normally requires your app to hold a secret credential. This
project supports one deployment mode that genuinely doesn't: running inside
Databricks itself, calling Databricks Foundation Model APIs.

```bash
LLM_PROVIDER=databricks
EMBEDDING_PROVIDER=databricks
# DATABRICKS_HOST / DATABRICKS_TOKEN can stay blank when this app runs
# inside a Databricks notebook, job, or App -- auth is then fully automatic.
```

With both switches set, `GEMINI_API_KEY` is never read (`config/settings.py`
only requires it when a provider is actually `"gemini"`) — Gemini is
completely out of the runtime path. `rag_pipeline/llm_service.py` and
`embeddings/embedding_service.py` are the two factories that dispatch on
these switches; nothing else in the app needs to change.

**Tested vs. not tested:** the provider-dispatch logic itself is covered by
`tests/test_provider_switching.py` (mocked, no real endpoint). Actually
calling a live Databricks Model Serving endpoint could **not** be tested in
this project's own build sandbox — there's no Databricks workspace attached
there. Smoke-test `LLM_PROVIDER=databricks` against your real workspace
before relying on it.

**Dependency note:** `requirements.txt` pins the older `langchain-databricks`
package rather than the newer `databricks-langchain`, because the newer one
pulls in `langchain-core` 1.x, which conflicts with `langchain-google-genai`
and with the MLflow version this project's local file-store tracking depends
on. If you go fully keyless (drop Gemini entirely), you can safely upgrade
to `databricks-langchain` and remove the Gemini-specific pins.

## 7. A note on this project's own development/test sandbox

This app was built and smoke-tested inside a network-restricted cloud
sandbox. Two sandbox-specific constraints are worth knowing about if you hit
them in a similar restricted environment (they do **not** apply on a normal
laptop/server with open internet access):

- **`huggingface.co` may be blocked** by an egress policy in some sandboxes.
  This app doesn't depend on it: there are no local models, so the only
  outbound hosts needed are PyPI (install time) and
  `generativelanguage.googleapis.com` (Gemini, at runtime).
- **No browser/port-forwarding was available** in that sandbox, so the
  Streamlit UI itself could only be smoke-tested for "does the process boot
  and serve HTTP" (`curl localhost:8501`), not a real interactive
  walkthrough. A live, interactive test of the chat UI should be done on
  your own machine or any environment with normal port access.
- Gemini model names move fast. If `GEMINI_CHAT_MODEL` or
  `GEMINI_EMBEDDING_MODEL` ever 404s with a "no longer available" message,
  call `GET https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY`
  to see current model names and update `.env`.
- The Gemini free tier caps embedding calls at roughly 100/minute, and each
  text in a batch counts individually against that quota. `scripts/ingest.py`
  paces itself under this limit automatically when `EMBEDDING_PROVIDER=gemini`
  (see `embeddings/gemini_embeddings.py`'s rate limiter) — ingesting a few
  hundred chunks may take several minutes on the free tier.

## 8. Testing

```bash
pytest
```

The full suite (`tests/`) runs offline and deterministically: `conftest.py`
swaps in a fake, hash-based embedding provider and points every on-disk
store (Chroma, SQLite, MLflow) at a fresh temp directory per test, and the
one test that reaches "the LLM" mocks `ChatGoogleGenerativeAI.invoke`
directly — no real API key or network call is exercised by the test suite.

## 9. Enhancements already built in

- **Conversational memory** (`rag_pipeline/memory.py`) — capped chat history for natural follow-ups.
- **Hybrid search** (`rag_pipeline/retrieval_service.py`) — vector + BM25 keyword blend, toggle via `ENABLE_HYBRID_SEARCH`.
- **Query rewriting** (`rag_pipeline/query_rewriter.py`) — follow-ups rewritten into standalone questions before retrieval.
- **Feedback logging** (`utils/feedback_logger.py`) — 👍/👎 buttons in the UI append to a local `.jsonl` file.
- **Guardrails** (`rag_pipeline/guardrails.py`) — prompt-injection denylist + lexical grounding check on every answer.
- **Source citations + confidence** — every answer shows an expandable source panel and a retrieval-based confidence score.
- **Follow-up handling** — covered jointly by memory + query rewriting above.
- **Multi-query retrieval** (`rag_pipeline/multi_query.py`) — searches with several LLM-generated paraphrasings of the question and fuses the results via reciprocal rank fusion, so retrieval isn't only as good as the user's exact wording. Off by default (`ENABLE_MULTI_QUERY`) — costs one extra Gemini chat call per question.
- **Multi-hop retrieval** (`rag_pipeline/multi_hop.py`) — after the first retrieval pass, lets the model ask itself a follow-up search query when a compound question needs a second, different piece of information, then merges both rounds' chunks before answering. Off by default (`ENABLE_MULTI_HOP`), bounded by `MAX_HOPS`.
- **Retrieval evaluation** (`utils/retrieval_metrics.py`, `scripts/evaluate_retrieval.py`) — Recall@K, Precision@K, NDCG@K, and MRR for the full deployed pipeline against hand-labeled ground truth — see section 5.
- **Streaming answers** (`rag_pipeline/rag_pipeline.py`'s `on_token` callback, used in `frontend/app.py`) — the answer renders token-by-token as Gemini generates it, instead of appearing all at once after the full response completes. Improves perceived latency only (same total generation time, same token cost); retrieval, grounding, memory, and MLflow logging are unaffected.
- **Concurrent multi-query retrieval** (`rag_pipeline.py`'s `retrieve()`) — when `ENABLE_MULTI_QUERY` is on, its independent per-variant searches run in a thread pool instead of one after another, cutting that feature's added latency roughly to the slowest single search instead of their sum.

## 10. Path to a real Databricks/production deployment

Nothing in this codebase needs to change structurally to move to Databricks
— only the modules explicitly called out in section 2's table get swapped.
The LLM and embedding swap (item 0 below) is already done — see section 6.

0. Chat + embeddings → set `LLM_PROVIDER=databricks` and
   `EMBEDDING_PROVIDER=databricks` (section 6). Already implemented and
   config-driven; no code change needed for this part.
1. `vector_store/chroma_manager.py` → a thin wrapper around a Databricks
   Vector Search endpoint + index (`databricks-vectorsearch` SDK), synced
   from a real Delta table.
2. `vector_store/metadata_table.py` → the SQLite calls become Delta table
   writes/reads via Spark or the Databricks SQL connector, using the exact
   same schema already defined here.
3. `config/settings.py` → read secrets via `dbutils.secrets.get(scope, key)`
   instead of `os.getenv`, since it's the only file that touches secrets.
4. `utils/mlflow_tracking.py` → change one line,
   `mlflow.set_tracking_uri(...)`, to point at the workspace-hosted tracking
   server instead of a local `mlruns/` folder; every `mlflow.log_*` call is
   unchanged.
5. Deploy `frontend/app.py` as a Databricks App (or any standard Streamlit
   host) instead of running it locally.

## 11. Project structure

```
rag_chatbot/
├── data/pdfs/                  # source PDFs (backend-managed, no upload UI)
├── ingestion/pdf_loader.py
├── chunking/                    # chunker.py (dispatch), structure_chunker.py (chapter/section-aware)
├── embeddings/                 # base.py, gemini_embeddings.py, databricks_embeddings.py, embedding_service.py
├── vector_store/                # chroma_manager.py, metadata_table.py
├── rag_pipeline/                # retrieval_service, multi_query, query_rewriter, multi_hop, memory, guardrails, prompt_templates, rag_pipeline
├── frontend/app.py              # Streamlit chat-only UI
├── config/settings.py
├── utils/                       # logging_utils, mlflow_tracking, feedback_logger
├── scripts/ingest.py            # CLI + auto-ingest entry point
├── scripts/evaluate_retrieval.py, eval_questions.py  # evaluation tooling
├── scripts/run_lan.ps1          # share on the local network (Windows)
├── Dockerfile                   # container image (company VM / sandbox)
├── utils/retrieval_metrics.py   # Recall@K, Precision@K, NDCG@K, MRR
├── tests/                       # pytest suite
├── conftest.py                  # shared fixtures (fake embeddings, temp stores)
├── requirements.txt
├── .env.example
└── README.md
```

## 12. Running on a company VM / sandbox

The code is on GitHub (private repo), so a Linux sandbox with outbound access
to GitHub, PyPI and `generativelanguage.googleapis.com` can run it directly:

```bash
git clone https://github.com/afaquzzafar/rag_chatbot_v2.git && cd rag_chatbot_v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then set GEMINI_API_KEY (and GEMINI_CHAT_MODEL)
python -m scripts.ingest
streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501
```

Or, where Docker is available, `Dockerfile` does all of the above:

```bash
docker build -t insurance-chatbot .
docker run -p 8501:8501 -e GEMINI_API_KEY=... -e GEMINI_CHAT_MODEL=... insurance-chatbot
```

Not carried over from GitHub (by design): `.env` (your key) and
`vectorstore_db/` (rebuilt by `scripts.ingest`). To see the UI, the sandbox
must expose port 8501 to your browser (a preview URL or port forwarding).

## 13. Sharing on your local network

To share the app from a Windows PC with other devices on the same
Wi-Fi/office network:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_lan.ps1
```

It prints the address to open on other devices (`http://<this-PC-IP>:8501`).
Windows Firewall must allow **inbound TCP 8501**. Either click **Allow** when
Windows asks the first time, or run once in an *administrator* PowerShell:

```powershell
New-NetFirewallRule -DisplayName "RAG chatbot (Streamlit 8501)" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow -Profile Private
```

The rule above only applies on networks marked **Private** (Settings ->
Network & internet -> Wi-Fi -> your network -> Private network). Only do that
for a network you trust, such as home or office. The app has no login, so
anyone on the network can use it and your Gemini quota. The PC must stay on
with the script running.
