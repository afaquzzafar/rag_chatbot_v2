---
# Hugging Face Space configuration (ignored everywhere else). The Space is
# built from ./Dockerfile; see "Deploying to Hugging Face Spaces" below.
title: Healthcare Insurance Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: RAG chatbot that answers from health plan documents
---

# Healthcare Insurance Assistant — Local RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers member, provider,
and policy questions using your official insurance plan documents (Evidence
of Coverage, Summary of Benefits, and related policy manuals). Built to run
entirely on a single machine — no cloud infrastructure required to try it out.

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
│  embeddings/embedding_service.py [pluggable: local (sentence-transformers)│
│        │                          or Gemini (gemini-embedding-001)]      │
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
│    reranker.py              optional cross-encoder re-scoring           │
│    query_rewriter.py        follow-up question -> standalone question   │
│    multi_hop.py             optional: follow-up retrieval round         │
│    memory.py                 capped conversation history                │
│    guardrails.py            input/output safety checks                  │
│    prompt_templates.py      grounded system prompt + citation format    │
│    rag_pipeline.py           RAGPipeline: retrieve→context→generate     │
│        │                                                                 │
│        ▼                                                                 │
│  Gemini Chat Model (gemini-3.8-flash)  ──► grounded, cited answer       │
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
| Azure OpenAI / `text-embedding-3-large` | **Pluggable provider**: local `sentence-transformers` (free/offline) or Gemini `gemini-embedding-001` | Matches the actual Gemini-only key you have; local is the zero-cost default, Gemini is the Phase 2 / sandbox-compatible option |
| MLflow on a Databricks tracking server | **MLflow, local file store** (`mlruns/`) | Identical `mlflow.log_*` API; only the tracking URI differs |

## 3. Setup (unrestricted machine — recommended path)

```bash
git clone <this repo>
cd rag_chatbot
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
# ^ installs a CPU-only build of torch (~3 GB total install) via the
#   --extra-index-url pin at the top of requirements.txt -- without it, pip
#   would pull a GPU/CUDA build (~7 GB) that this project never uses, since
#   local embeddings and reranking only ever run on CPU here.

cp .env.example .env
# Edit .env and set GEMINI_API_KEY (get one free at https://aistudio.google.com/apikey)
# Leave EMBEDDING_PROVIDER=local for a fully free embedding pipeline.

python -m scripts.ingest           # one-time: indexes data/pdfs/*.pdf
streamlit run frontend/app.py      # opens the chat UI in your browser
```

Add your own PDFs by dropping them into `data/pdfs/` and re-running
`python -m scripts.ingest` (or just restarting the Streamlit app — it
auto-detects new/changed files via a content hash and only processes what's
new).

## 4. Switching embedding providers (Phase 1 → Phase 2)

Everything routes through `embeddings/embedding_service.py`, so this is a
one-line change in `.env`:

```bash
EMBEDDING_PROVIDER=local     # free, offline, sentence-transformers (default)
EMBEDDING_PROVIDER=gemini    # hosted, uses your GEMINI_API_KEY, no model download
```

**Important:** switching providers changes the vector space. Re-run
`python -m scripts.ingest --force` equivalent (or simply delete
`vectorstore_db/` and re-run `python -m scripts.ingest`) after switching —
old vectors from one provider are not comparable to query vectors from the
other.

## 5. Choosing which local embedding model to use

`LOCAL_EMBEDDING_MODEL` defaults to `all-MiniLM-L6-v2` — the smallest and
fastest free CPU embedding model, chosen for "run locally as easily as
possible." It is not the most accurate option available. Rather than trust
a public leaderboard (which measures general web/Wikipedia text, not
insurance-specific language), this project includes its own small
MTEB-style evaluation tool that measures retrieval quality against your
**actual** ingested documents:

```bash
python -m scripts.evaluate_embeddings
```

This downloads a handful of candidate models (see `CANDIDATE_MODELS` at the
top of `scripts/evaluate_embeddings.py` — edit that list to try others),
embeds your real chunks and a set of hand-labeled real questions
(`EVAL_QUESTIONS`, shared from `scripts/eval_questions.py`), and reports
Recall@1/3/5, Precision@1/3/5, NDCG@1/3/5, and MRR per model, ending with a
recommendation and the exact `.env` line to apply it. Needs
`huggingface.co` reachable (won't run in the network-restricted sandbox
this project was partly built in — see section 7). After switching models,
re-ingest as described above. Output is printed only, not saved anywhere.

**Important:** this evaluates raw embedding-model similarity only — no
hybrid search, reranking, or score threshold, so it doesn't reflect what a
real user actually gets back. To evaluate the full, deployed retrieval
pipeline (the one real questions actually go through) against the same
ground truth, run:

```bash
python -m scripts.evaluate_retrieval   # needs the vector store already populated
```

This uses whichever provider/settings are currently in `.env`, reports the
same four metrics **twice — once BEFORE reranking (vector/hybrid search
only) and once AFTER** — plus a one-line summary of how many questions
reranking actually improved, left unchanged, or made worse. That's the only
way to see reranking's real effect: `RAGPipeline.retrieve()` on its own only
ever returns the final, post-reranking list, so without this the
pre-reranking ranking is invisible. If `ENABLE_RERANKING=false`, the two
stages are identical (nothing to compare) and the impact summary is
skipped.

Unlike the embedding-model comparison above, this also saves every
individual question's before/after result (not just the average) to
`eval_results/retrieval_eval_<timestamp>.json`, so a specific regression, or
reranking actively hurting one particular question, can be traced back by
name. It also prints which questions missed the correct page within
`TOP_K` after reranking, for quick debugging. The underlying metric
formulas live in `utils/retrieval_metrics.py`, shared by
both scripts so their numbers are directly comparable.

**Reviewing metrics in git, without CI:** every run also overwrites
`eval_results/latest.json` — the one snapshot in that folder that ISN'T
gitignored, specifically so it can be committed and diffed in a PR like any
other file. It also records a `corpus_fingerprint` (a hash of every file in
`data/pdfs/`) plus the active `embedding_model` and `reranker_model`, so
anyone reviewing a commit of this file can tell at a glance whether a change
in the numbers was caused by different documents or a different model. This
is deliberately a manual step, not automatic: re-run
`python -m scripts.evaluate_retrieval` and commit `eval_results/latest.json`
whenever you change `data/pdfs/`, `RERANKER_MODEL`, or the embedding
provider/model, the same way you'd update any other generated file that
depends on project state. Wiring an actual CI trigger for this (so it
happens on every relevant PR rather than being something you remember to
do) is a reasonable next step, but is a separate, bigger change (needs
`GEMINI_API_KEY` as a repo secret and network access in CI) that hasn't been
built here.

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
  The `local` embedding provider downloads its model from the Hugging Face
  Hub on first use, so it will fail there — use `EMBEDDING_PROVIDER=gemini`
  instead in that environment. The `reranker` (cross-encoder) has the same
  dependency and degrades gracefully (logs a warning, skips reranking)
  rather than crashing if it can't download its model.
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
  hundred chunks may take several minutes on the free tier. This does not
  apply to the `local` provider, which has no API rate limit.

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
- **Reranking** (`rag_pipeline/reranker.py`) — optional CPU cross-encoder re-scoring, toggle via `ENABLE_RERANKING`.
- **Feedback logging** (`utils/feedback_logger.py`) — 👍/👎 buttons in the UI append to a local `.jsonl` file.
- **Guardrails** (`rag_pipeline/guardrails.py`) — prompt-injection denylist + lexical grounding check on every answer.
- **Source citations + confidence** — every answer shows an expandable source panel and a retrieval-based confidence score.
- **Follow-up handling** — covered jointly by memory + query rewriting above.
- **Multi-query retrieval** (`rag_pipeline/multi_query.py`) — searches with several LLM-generated paraphrasings of the question and fuses the results via reciprocal rank fusion, so retrieval isn't only as good as the user's exact wording. Off by default (`ENABLE_MULTI_QUERY`) — costs one extra Gemini chat call per question.
- **Multi-hop retrieval** (`rag_pipeline/multi_hop.py`) — after the first retrieval pass, lets the model ask itself a follow-up search query when a compound question needs a second, different piece of information, then merges both rounds' chunks before answering. Off by default (`ENABLE_MULTI_HOP`), bounded by `MAX_HOPS`.
- **Retrieval evaluation** (`utils/retrieval_metrics.py`, `scripts/evaluate_embeddings.py`, `scripts/evaluate_retrieval.py`) — Recall@K, Precision@K, NDCG@K, and MRR against hand-labeled ground truth, for both a candidate embedding model in isolation and the full deployed pipeline — see section 5.
- **Streaming answers** (`rag_pipeline/rag_pipeline.py`'s `on_token` callback, used in `frontend/app.py`) — the answer renders token-by-token as Gemini generates it, instead of appearing all at once after the full response completes. Improves perceived latency only (same total generation time, same token cost); retrieval, grounding, memory, and MLflow logging are unaffected.
- **Concurrent multi-query retrieval** (`rag_pipeline.py`'s `retrieve_with_stages()`) — when `ENABLE_MULTI_QUERY` is on, its independent per-variant searches run in a thread pool instead of one after another, cutting that feature's added latency roughly to the slowest single search instead of their sum.

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
├── embeddings/                 # base.py, local_embeddings.py, gemini_embeddings.py, embedding_service.py
├── vector_store/                # chroma_manager.py, metadata_table.py
├── rag_pipeline/                # retrieval_service, multi_query, reranker, query_rewriter, multi_hop, memory, guardrails, prompt_templates, rag_pipeline
├── frontend/app.py              # Streamlit chat-only UI
├── config/settings.py
├── utils/                       # logging_utils, mlflow_tracking, feedback_logger
├── scripts/ingest.py            # CLI + auto-ingest entry point
├── scripts/evaluate_embeddings.py, evaluate_retrieval.py, eval_questions.py  # evaluation tooling
├── utils/retrieval_metrics.py   # Recall@K, Precision@K, NDCG@K, MRR
├── tests/                       # pytest suite
├── conftest.py                  # shared fixtures (fake embeddings, temp stores)
├── requirements.txt
├── .env.example
└── README.md
```

## 12. Deploying to Hugging Face Spaces

The repo is ready to run as a Hugging Face **Docker Space**: the YAML header at
the top of this README configures the Space, and `Dockerfile` builds an image
with the dependencies, both local models and the vector index baked in (so
the app starts fast and needs only `GEMINI_API_KEY` at runtime).

```bash
hf auth login                                   # once; paste a HF *write* token
python -m scripts.deploy_hf_space <hf-user>/healthcare-insurance-assistant
```

`scripts/deploy_hf_space.py` creates the Space (private by default; add
`--public` to share it), sets the non-secret settings (chat model,
multi-query / multi-hop flags) as Space variables, and uploads the project,
excluding `.env` and local runtime data. Then, in the Space's **Settings ->
Variables and secrets**, add `GEMINI_API_KEY` as a **secret**. Re-run the same
command to deploy code changes.

Notes:
- The hosted app has no login of its own. A **private** Space is only usable by
  you (and members you add); a **public** Space lets anyone spend your Gemini
  quota.
- Space storage is not persistent: feedback (`vectorstore_db/feedback.jsonl`)
  and MLflow runs are lost when the Space restarts. The index itself is
  rebuilt into the image on every deploy.

## 13. Sharing on your local network

Hosting on Hugging Face now needs a paid PRO plan for Docker Spaces, so the
free way to share the app is from this PC to other devices on the same
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
