# ==============================================================================
# utils/retrieval_metrics.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Standard information-retrieval metrics, computed by hand from a simple,
#   shared representation: the 1-indexed ranks at which the RELEVANT item(s)
#   for one query were found among the ranked results (best match first).
#   An empty list means "not found at all" (within however many results were
#   considered).
#
# WHY THIS SHAPE, SPECIFICALLY
#   scripts/evaluate_retrieval.py (evaluates the full, deployed retrieval
#   pipeline) needs these four standard metrics computed exactly and
#   consistently. Factoring the math out here keeps it separately testable
#   and reusable by any future evaluation script.
#
# THE FOUR METRICS
#   - Recall@K:    of all truly relevant items, what fraction were found
#                  within the top K? With this project's ground truth (one
#                  known-correct page per question), this is equivalent to a
#                  simple hit-rate: 1.0 if that page appears in the top K,
#                  else 0.0.
#   - Precision@K: of the K items actually returned, what fraction were
#                  relevant? With a single relevant page per question, this
#                  is naturally small (at most 1/K) -- that's expected, not a
#                  bug: precision penalizes a system for surfacing K results
#                  when only one of them could ever be right.
#   - MRR:         1 / rank of the first relevant item (0 if never found).
#                  Rewards getting the right answer NEAR the top, not just
#                  somewhere in a long list -- the standard primary metric
#                  for comparing retrieval quality.
#   - NDCG@K:      like Recall@K, but rank-sensitive: a relevant item at rank
#                  1 scores higher than the same item at rank 5. Normalized
#                  against the best possible ordering (IDCG) so the score is
#                  always in [0, 1] regardless of K or how many relevant
#                  items exist.
#
# INPUT / OUTPUT
#   Each function takes `relevant_ranks: List[int]` -- the 1-indexed ranks of
#   every relevant item actually found (usually just one, for this project's
#   single-correct-page ground truth) -- plus `k` and, where the formula
#   needs it, `num_relevant` (the total number of truly relevant items that
#   exist for this query, not just how many were found).
# ==============================================================================

import math
from typing import List


def recall_at_k(relevant_ranks: List[int], k: int, num_relevant: int) -> float:
    """Fraction of all relevant items that were found within the top k."""
    if num_relevant <= 0:
        return 0.0
    hits = sum(1 for rank in relevant_ranks if rank <= k)
    return hits / num_relevant


def precision_at_k(relevant_ranks: List[int], k: int) -> float:
    """Fraction of the top k results that were actually relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for rank in relevant_ranks if rank <= k)
    return hits / k


def reciprocal_rank(relevant_ranks: List[int]) -> float:
    """1 / rank of the FIRST relevant item found, or 0.0 if none were found."""
    if not relevant_ranks:
        return 0.0
    return 1.0 / min(relevant_ranks)


def ndcg_at_k(relevant_ranks: List[int], k: int, num_relevant: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k, for binary relevance.

    DCG@k rewards a relevant item more the earlier it appears (1/log2(rank+1)
    per hit); IDCG@k is the DCG of the best possible ordering (every relevant
    item packed into the first `num_relevant` positions). Dividing by IDCG
    keeps the score in [0, 1] so it's comparable across queries with
    different numbers of relevant items.
    """
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks if rank <= k)

    ideal_hit_count = min(num_relevant, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hit_count + 1))

    return dcg / idcg if idcg > 0 else 0.0
