# ==============================================================================
# scripts/evaluate_retrieval.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Evaluates the FULL, DEPLOYED retrieval pipeline (RAGPipeline: vector/
#   hybrid search + optional multi-query + score threshold, using whatever
#   provider and settings are currently configured in .env) against
#   hand-labeled ground truth (scripts/eval_questions.py) -- the end-to-end
#   retrieval quality a real user actually gets back right now.
#
# METRICS: Recall@K, Precision@K, NDCG@K, MRR
#   All four computed by utils/retrieval_metrics.py -- see that module for
#   the exact formulas. Every question here has exactly one known-correct
#   (file, page), so num_relevant=1 throughout.
#
# WHAT GETS SAVED, AND WHERE
#   ONE JSON file per run to eval_results/, containing the aggregate metrics
#   AND every individual question's result -- so a specific regression can
#   be traced back by name.
#
# PREREQUISITE
#   The vector store must already be populated: run `python -m scripts.ingest`
#   first. This script does not build embeddings itself -- it queries
#   whatever is already indexed, exactly like a real user's question would.
#
# HOW TO RUN THIS
#   python -m scripts.evaluate_retrieval
# ==============================================================================

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from config.settings import settings
from rag_pipeline.rag_pipeline import RAGPipeline
from rag_pipeline.retrieval_service import RetrievedChunk
from scripts.eval_questions import EVAL_QUESTIONS, EvalQuestion
from utils.logging_utils import get_logger
from utils.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

logger = get_logger(__name__)

RESULTS_DIR = Path("eval_results")
LATEST_RESULTS_PATH = RESULTS_DIR / "latest.json"  # the one snapshot meant to be committed -- see .gitignore
K_VALUES = (1, 3, 5)

_ACTIVE_EMBEDDING_MODEL = {
    "gemini": lambda: settings.gemini_embedding_model,
    "databricks": lambda: settings.databricks_embedding_endpoint,
}


def _corpus_fingerprint(pdf_dir: str) -> str:
    """
    A single hash summarizing every source PDF's current content.

    Recorded alongside the metrics so a committed eval_results/latest.json
    makes it visible, on review, exactly which document set + model
    configuration produced these numbers -- if this fingerprint differs
    between two commits of latest.json, the documents changed; if it's the
    same but the embedding model fields differ, that's what changed
    instead.
    """
    folder = Path(pdf_dir)
    if not folder.exists():
        return ""
    combined = hashlib.sha256()
    for pdf_path in sorted(folder.glob("*.pdf")):
        combined.update(pdf_path.name.encode("utf-8"))
        combined.update(pdf_path.read_bytes())
    return combined.hexdigest()[:16]


@dataclass
class StageMetrics:
    """Every metric for one question's retrieved chunks."""

    found_rank: Optional[int]  # None if the correct page was never retrieved
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    reciprocal_rank: float


@dataclass
class QueryResult:
    """One question's results, kept individually."""

    question: str
    expected_file: str
    expected_page: int
    metrics: StageMetrics
    retrieved: List[Dict[str, Any]]


def _find_rank(chunks: List[RetrievedChunk], eval_q: EvalQuestion) -> Optional[int]:
    """1-indexed rank of the first retrieved chunk matching the correct (file, page)."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.document_name == eval_q.expected_file and chunk.page_number == eval_q.expected_page:
            return rank
    return None


def _stage_metrics(chunks: List[RetrievedChunk], eval_q: EvalQuestion) -> StageMetrics:
    found_rank = _find_rank(chunks, eval_q)
    # Exactly one relevant page per question -- see utils/retrieval_metrics.py.
    relevant_ranks = [found_rank] if found_rank is not None else []
    return StageMetrics(
        found_rank=found_rank,
        recall_at_k={k: recall_at_k(relevant_ranks, k, num_relevant=1) for k in K_VALUES},
        precision_at_k={k: precision_at_k(relevant_ranks, k) for k in K_VALUES},
        ndcg_at_k={k: ndcg_at_k(relevant_ranks, k, num_relevant=1) for k in K_VALUES},
        reciprocal_rank=reciprocal_rank(relevant_ranks),
    )


def _evaluate_one(pipeline: RAGPipeline, eval_q: EvalQuestion) -> QueryResult:
    chunks = pipeline.retrieve(eval_q.question)
    return QueryResult(
        question=eval_q.question,
        expected_file=eval_q.expected_file,
        expected_page=eval_q.expected_page,
        metrics=_stage_metrics(chunks, eval_q),
        retrieved=[
            {"document_name": c.document_name, "page_number": c.page_number, "score": round(c.score, 4)}
            for c in chunks
        ],
    )


def _aggregate_stage(stage_results: List[StageMetrics]) -> Dict[str, Any]:
    return {
        "recall_at_k": {k: mean(s.recall_at_k[k] for s in stage_results) for k in K_VALUES},
        "precision_at_k": {k: mean(s.precision_at_k[k] for s in stage_results) for k in K_VALUES},
        "ndcg_at_k": {k: mean(s.ndcg_at_k[k] for s in stage_results) for k in K_VALUES},
        "mrr": mean(s.reciprocal_rank for s in stage_results),
    }


def _print_stage(aggregate: Dict[str, Any]) -> None:
    for k in K_VALUES:
        print(
            f"Recall@{k}={aggregate['recall_at_k'][k]:.0%}   "
            f"Precision@{k}={aggregate['precision_at_k'][k]:.2f}   "
            f"NDCG@{k}={aggregate['ndcg_at_k'][k]:.3f}"
        )
    print(f"MRR={aggregate['mrr']:.3f}")


def main() -> None:
    print(
        f"Evaluating the full retrieval pipeline against {len(EVAL_QUESTIONS)} questions "
        f"(embedding_provider={settings.embedding_provider}, "
        f"hybrid_search={settings.enable_hybrid_search}, multi_query={settings.enable_multi_query})...\n"
        "(Requires the vector store to already be populated -- run "
        "`python -m scripts.ingest` first if you haven't.)\n"
    )

    pipeline = RAGPipeline()
    results = [_evaluate_one(pipeline, eval_q) for eval_q in EVAL_QUESTIONS]
    aggregate = _aggregate_stage([r.metrics for r in results])

    print("=" * 70)
    _print_stage(aggregate)
    print("=" * 70)

    missed = [r for r in results if r.metrics.found_rank is None]
    if missed:
        print(f"\n{len(missed)} question(s) missed the correct page within top {settings.top_k}:")
        for r in missed:
            print(f"  - \"{r.question}\" (expected {r.expected_file} p.{r.expected_page})")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": timestamp,
        "settings": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": _ACTIVE_EMBEDDING_MODEL[settings.embedding_provider](),
            "enable_hybrid_search": settings.enable_hybrid_search,
            "enable_multi_query": settings.enable_multi_query,
            "top_k": settings.top_k,
            "score_threshold": settings.score_threshold,
        },
        "corpus_fingerprint": _corpus_fingerprint(settings.pdf_data_dir),
        "aggregate": aggregate,
        "per_query": [asdict(r) for r in results],
    }
    serialized = json.dumps(payload, indent=2)

    # Timestamped copy: local, disposable run history (gitignored).
    (RESULTS_DIR / f"retrieval_eval_{timestamp}.json").write_text(serialized)

    # latest.json: the one snapshot meant to be committed and reviewed in
    # git -- see .gitignore. It's just overwritten on every run; nothing
    # forces it to be re-run automatically, so it's only ever as fresh as
    # the last time someone deliberately ran this script and committed the
    # change (e.g. after editing data/pdfs/ or EMBEDDING_PROVIDER).
    LATEST_RESULTS_PATH.write_text(serialized)

    print(f"\nFull per-query results saved to {RESULTS_DIR / f'retrieval_eval_{timestamp}.json'}")
    print(f"Committed snapshot updated at {LATEST_RESULTS_PATH} -- `git add` and commit it to record this run.")


if __name__ == "__main__":
    main()
