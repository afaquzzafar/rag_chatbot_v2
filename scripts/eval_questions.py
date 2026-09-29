# ==============================================================================
# scripts/eval_questions.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   The ground-truth question set used by evaluate_retrieval.py (evaluates
#   the full, deployed retrieval pipeline). Kept in its own module so any
#   future evaluation script is measured against the identical questions --
#   otherwise their numbers wouldn't be comparable.
#
# WHERE THIS GROUND TRUTH CAME FROM
#   Each row is a realistic member question, hand-paired with the
#   (file_name, page_number) that actually answers it -- verified by hand
#   against data/pdfs/*.pdf. Add more rows here whenever a new source
#   document is added; more questions make both evaluations more reliable.
# ==============================================================================

from dataclasses import dataclass
from typing import List


@dataclass
class EvalQuestion:
    """One test question, paired with the exact source page that answers it."""

    question: str
    expected_file: str
    expected_page: int


EVAL_QUESTIONS: List[EvalQuestion] = [
    EvalQuestion("What is the annual medical deductible?", "Summary_of_Benefits.pdf", 1),
    EvalQuestion("What is the out-of-pocket maximum for a family?", "Summary_of_Benefits.pdf", 1),
    EvalQuestion("How much is a primary care visit copay?", "Summary_of_Benefits.pdf", 2),
    EvalQuestion("What do I pay for an emergency room visit?", "Summary_of_Benefits.pdf", 2),
    EvalQuestion("How much does a generic Tier 1 drug cost?", "Summary_of_Benefits.pdf", 3),
    EvalQuestion("Is routine dental cleaning covered?", "Summary_of_Benefits.pdf", 4),
    EvalQuestion("How many eye exams are covered per year?", "Summary_of_Benefits.pdf", 4),
    EvalQuestion("What is the definition of coinsurance?", "Evidence_of_Coverage.pdf", 64),
    EvalQuestion("Are emergency services covered outside the network?", "Evidence_of_Coverage.pdf", 19),
    EvalQuestion("What services are excluded from coverage?", "Evidence_of_Coverage.pdf", 26),
    EvalQuestion("How many days do I have to file an appeal?", "Evidence_of_Coverage.pdf", 44),
    EvalQuestion("What is prior authorization and when do I need it?", "Evidence_of_Coverage.pdf", 16),
    EvalQuestion("Does the deductible apply to preventive care visits?", "Evidence_of_Coverage.pdf", 24),
    EvalQuestion("Can I add my spouse to my health plan?", "Policy_Manual.pdf", 1),
    EvalQuestion("What is coordination of benefits?", "Policy_Manual.pdf", 3),
    EvalQuestion("How long does a standard prior authorization review take?", "Policy_Manual.pdf", 4),
    EvalQuestion("What are the two levels of the appeals process?", "Policy_Manual.pdf", 5),
    EvalQuestion("How are network providers credentialed?", "Policy_Manual.pdf", 6),
    EvalQuestion("How do I report suspected insurance fraud?", "Policy_Manual.pdf", 8),
    EvalQuestion("How can I request a copy of my health records?", "Policy_Manual.pdf", 9),
]
