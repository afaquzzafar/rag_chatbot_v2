# ==============================================================================
# config/settings.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Centralizes every tunable value in the app (API keys, model names, file
#   paths, chunk sizes, retrieval knobs) in ONE typed object called `settings`.
#   Every other module imports `settings` instead of calling os.getenv()
#   directly, so there is exactly one place to look when you want to change
#   how the app behaves.
#
# WHY THIS MATTERS (GenAI beginners)
#   A RAG app has many moving parts you'll want to tune while experimenting:
#   which LLM answers questions, which embedding model turns text into
#   vectors, how big each chunk is, how many chunks to retrieve, etc.
#   Keeping them in a dataclass populated from environment variables means:
#     - secrets (API keys) never get hard-coded into source files
#     - you can change behavior by editing `.env`, no code changes needed
#     - it's obvious at a glance what the app depends on
#
# DATABRICKS CONCEPT MAPPING (for the "production" reader)
#   In a Databricks deployment, secrets normally live in a Databricks Secret
#   Scope (`dbutils.secrets.get(scope, key)`) instead of a local `.env` file,
#   and paths would point at Unity Catalog volumes / Delta tables instead of
#   a local folder. This file isolates that boundary: only this module reads
#   `os.getenv`, so swapping the local `.env` source for `dbutils.secrets`
#   later means editing one file, not the whole codebase.
# ==============================================================================

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# load_dotenv() reads a ".env" file (if present) in the project root and
# copies its KEY=VALUE lines into the process environment (os.environ).
# This is how we keep the Gemini API key out of source control: it lives
# in a local .env file (gitignored) that this call picks up at startup.
load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    """Parse an environment variable as a boolean ('true'/'1'/'yes' -> True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """
    Typed, single source of truth for all application configuration.

    Grouped by concern (LLM, embeddings, vector store, chunking, retrieval,
    observability) so it reads like a table of contents for the whole app.
    """

    # -- LLM provider (pluggable: "gemini" needs an API key; "databricks" is ---
    # -- fully keyless when run inside a Databricks workspace/app) -----------
    # This is the same factory-pattern switch used for embeddings below.
    # "databricks" routes chat calls to a Databricks Model Serving endpoint
    # (Foundation Model APIs) via the Databricks SDK's own auth chain --
    # inside a Databricks notebook, job, or App, that auth is automatic
    # (no key of any kind lives in this app's config or code at all). Outside
    # Databricks, DATABRICKS_HOST + DATABRICKS_TOKEN below act as an explicit
    # fallback credential for local testing against a real workspace.
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "databricks"

    # -- Gemini (used when LLM_PROVIDER=gemini and/or EMBEDDING_PROVIDER=gemini)
    # GEMINI_API_KEY is required for either. Get one at https://aistudio.google.com/apikey
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    # Gemini's free tier caps CHAT generation at a much stricter ~5
    # requests/minute (separate from the embedding quota above). Both the
    # chat call and the query-rewrite call draw from this same quota, so
    # asking 2-3 questions in a row can exhaust it -- this paces both
    # proactively instead of letting the app fail and retry blindly.
    gemini_chat_requests_per_minute: int = int(os.getenv("GEMINI_CHAT_REQUESTS_PER_MINUTE", "5"))

    # -- Databricks (used when LLM_PROVIDER=databricks and/or --------------------
    # -- EMBEDDING_PROVIDER=databricks) -------------------------------------------
    # DATABRICKS_HOST/DATABRICKS_TOKEN are only needed when calling a
    # workspace from OUTSIDE it (e.g. this app running on your laptop against
    # a remote workspace). Leave them blank when this app itself runs inside
    # Databricks (a notebook, job, or Databricks App) -- the SDK picks up the
    # ambient workspace identity automatically, which is what makes that
    # deployment mode genuinely keyless.
    databricks_host: str = os.getenv("DATABRICKS_HOST", "")
    databricks_token: str = os.getenv("DATABRICKS_TOKEN", "")
    databricks_llm_endpoint: str = os.getenv("DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
    databricks_embedding_endpoint: str = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en")
    databricks_temperature: float = float(os.getenv("DATABRICKS_TEMPERATURE", "0.1"))

    # -- Embeddings (pluggable: "local", "gemini", or "databricks") ----------
    # This single switch is the whole point of the abstraction in
    # embeddings/: change EMBEDDING_PROVIDER once you're ready to move off
    # free local embeddings, and nothing else in the app changes.
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")  # "local" | "gemini" | "databricks"
    # Chosen empirically, not by default: scripts/evaluate_embeddings.py's
    # MTEB-style evaluation (Recall@K + MRR against this project's own real
    # documents and 20 hand-verified questions) measured all-mpnet-base-v2
    # at MRR=0.950, clearly ahead of the original all-MiniLM-L6-v2 default.
    # It's a larger model (~420MB vs. ~80MB) and somewhat slower to embed
    # with, but for a single-user local app that trade favors accuracy. If
    # you add substantially different documents later, re-run that
    # evaluation -- the best model for one corpus isn't guaranteed to stay
    # best for another.
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    # Gemini's free tier caps embed_content at ~100 requests/minute, and each
    # text embedded (even inside one batched call) counts as one request
    # against that quota. Default to a safety margin below the documented
    # limit; raise this once you're on a paid tier with higher quota.
    gemini_embedding_requests_per_minute: int = int(
        os.getenv("GEMINI_EMBEDDING_REQUESTS_PER_MINUTE", "90")
    )

    # -- Vector store (ChromaDB stands in for Databricks Vector Search) ------
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "vectorstore_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "insurance_docs")

    # -- Metadata store (SQLite stands in for a Databricks Delta Table) ------
    metadata_db_path: str = os.getenv("METADATA_DB_PATH", "vectorstore_db/metadata.sqlite")

    # -- Document ingestion ---------------------------------------------------
    pdf_data_dir: str = os.getenv("PDF_DATA_DIR", "data/pdfs")

    # -- Chunking ---------------------------------------------------------------
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # -- Retrieval ----------------------------------------------------------------
    top_k: int = int(os.getenv("TOP_K", "5"))
    # A threshold of 0.0 filters nothing, which means the app would always
    # show exactly top_k "sources" even when none of them are actually
    # relevant -- a fixed-size list of citations, some scoring ~0, is
    # actively misleading in an insurance context. 0.3 is a deliberately
    # moderate default: high enough to drop obviously irrelevant chunks,
    # low enough not to hide a genuinely useful but imperfect match. Tune
    # this against your own retrieved-score distribution once you have
    # real usage data; there is nothing universal about 0.3 itself.
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.3"))
    enable_hybrid_search: bool = _bool_env("ENABLE_HYBRID_SEARCH", True)
    enable_reranking: bool = _bool_env("ENABLE_RERANKING", True)
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # The ms-marco cross-encoder is trained on specific question -> passage
    # matches, so broad questions ("key points of coverage", "what are my
    # benefits?") and two-part questions score near 0 against EVERY chunk,
    # and the post-rerank threshold then drops all of them -> "not found"
    # even though relevant pages exist. When that happens, pass the best
    # RERANK_FALLBACK_K reranked chunks to the LLM anyway: the system prompt
    # still forces the exact "not found" sentence if they don't contain the
    # answer, and the (low) confidence shown reflects the weak match.
    # 0 restores the strict behavior (threshold can empty the result).
    rerank_fallback_k: int = int(os.getenv("RERANK_FALLBACK_K", "3"))

    # -- Multi-query retrieval (rag_pipeline/multi_query.py) ------------------
    # Off by default: unlike hybrid search and reranking (both local, free),
    # this costs one extra Gemini CHAT call per question -- against the free
    # tier's tight 5 requests/minute chat quota (shared with query rewriting
    # and answer generation), that adds up fast. Turn on once you're past the
    # free tier, or don't mind slower responses.
    enable_multi_query: bool = _bool_env("ENABLE_MULTI_QUERY", False)
    multi_query_variants: int = int(os.getenv("MULTI_QUERY_VARIANTS", "3"))

    # -- Multi-hop retrieval (rag_pipeline/multi_hop.py) ----------------------
    # Also off by default, and for the same reason: each extra hop is one
    # more chat call (the sufficiency check) plus one more retrieval pass.
    # max_hops bounds it (2 = one initial retrieval + at most one follow-up)
    # so a model that's "never quite satisfied" can't loop indefinitely.
    enable_multi_hop: bool = _bool_env("ENABLE_MULTI_HOP", False)
    max_hops: int = int(os.getenv("MAX_HOPS", "2"))

    # -- Conversation memory -----------------------------------------------------
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))

    # -- Observability --------------------------------------------------------------
    mlflow_tracking_dir: str = os.getenv("MLFLOW_TRACKING_DIR", "mlruns")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "insurance-rag-chatbot")
    feedback_log_path: str = os.getenv("FEEDBACK_LOG_PATH", "vectorstore_db/feedback.jsonl")

    # -- Logging --------------------------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        """
        Fail loudly and early if required config is missing.

        Why raise here instead of letting the first LLM call fail with a
        cryptic 401? A missing credential is a *configuration* problem the
        developer should fix before anything else runs, not a runtime error
        buried three function calls deep.

        Note what is deliberately NOT validated here: DATABRICKS_HOST /
        DATABRICKS_TOKEN are never required, even when LLM_PROVIDER or
        EMBEDDING_PROVIDER is "databricks" -- inside a real Databricks
        notebook, job, or App, the Databricks SDK authenticates from the
        ambient workspace identity with no env vars set at all. Requiring
        them here would break that genuinely keyless deployment mode.
        """
        if self.llm_provider not in {"gemini", "databricks"}:
            raise ValueError(
                f"LLM_PROVIDER must be 'gemini' or 'databricks', got '{self.llm_provider}'"
            )
        if self.embedding_provider not in {"local", "gemini", "databricks"}:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be 'local', 'gemini', or 'databricks', got "
                f"'{self.embedding_provider}'"
            )
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "your key from https://aistudio.google.com/apikey -- or set "
                "LLM_PROVIDER=databricks to run keyless on Databricks instead."
            )
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY to be set.")


# Module-level singleton: every other file does `from config.settings import settings`.
settings = Settings()
