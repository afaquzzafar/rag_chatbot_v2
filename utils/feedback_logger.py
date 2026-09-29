# ==============================================================================
# utils/feedback_logger.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Appends a single JSON line per user feedback event (thumbs up / down on
#   an answer) to a local .jsonl file. This is the "Feedback Logging"
#   enhancement from the spec.
#
# WHY THIS MATTERS
#   Retrieval-augmented answers can be subtly wrong even when every chunk is
#   grounded (e.g. the right chunk retrieved, but the model summarized it
#   poorly). The only reliable signal for "is this actually helping members"
#   is real user feedback. Logging it -- question, answer, sources, rating --
#   gives you a dataset to later use for prompt tuning, or retrieval tuning
#   (e.g. picking a better score threshold).
#
# WHY JSON LINES (.jsonl) INSTEAD OF SQLITE HERE
#   Feedback is write-heavy and append-only, and rarely needs SQL-style
#   querying inside the app itself -- a plain file you can `tail -f` or load
#   into a notebook with `pandas.read_json(path, lines=True)` is simpler and
#   has zero schema-migration overhead. Contrast with vector_store/
#   metadata_table.py, which genuinely needs indexed lookups and uses SQLite.
#
# INPUT / OUTPUT
#   Input:  question, answer, list of source citations, a rating ("up"/"down"),
#           optional free-text comment.
#   Output: one line appended to `settings.feedback_log_path`.
# ==============================================================================

import json
import os
import time
from typing import Any, Dict, List, Optional

from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def log_feedback(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    rating: str,
    comment: Optional[str] = None,
) -> None:
    """
    Append one feedback record to the local feedback log.

    Args:
        question: the user's original question.
        answer:   the assistant's answer being rated.
        sources:  the citation metadata shown alongside that answer.
        rating:   "up" or "down".
        comment:  optional free-text detail from the user.
    """
    os.makedirs(os.path.dirname(settings.feedback_log_path), exist_ok=True)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": sources,
        "rating": rating,
        "comment": comment,
    }

    try:
        with open(settings.feedback_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Logged '%s' feedback for question: %.60s...", rating, question)
    except OSError:
        # Feedback logging should never crash the chat experience -- a
        # failed write here is annoying but not user-facing.
        logger.exception("Failed to write feedback record")
