# ==============================================================================
# rag_pipeline/rag_pipeline.py
# ------------------------------------------------------------------------------
# THE CENTRAL ORCHESTRATOR: wires retrieval, guardrails, prompting,
# and the Gemini chat model into one class the Streamlit frontend calls.
#
# WHAT THIS FILE DOES
#   Defines RAGPipeline, with four methods matching the spec's required
#   method names -- each one a single, clear stage of "Retrieve Top K
#   Chunks -> Ground LLM Response Using Retrieved Context -> Return Answer
#   with Citations":
#
#     retrieve(question)        Step: vector (+hybrid+multi-query) search
#     build_context(chunks)     Step: format chunks into a citation-ready
#                                block of text for the prompt
#     generate_answer(...)      Step: call Gemini with the grounded prompt
#     answer_question(question) The single public entry point that runs the
#                                whole pipeline end-to-end for one turn,
#                                including the optional multi-hop follow-up
#                                round between build_context and generate_answer
#
# WHY SPLIT INTO FOUR METHODS INSTEAD OF ONE BIG FUNCTION
#   Each stage is independently testable (see tests/test_rag_pipeline.py,
#   which tests build_context's formatting without calling any LLM), and
#   independently reusable -- e.g. a future admin "debug this retrieval"
#   tool could call retrieve() + build_context() without ever generating an
#   answer.
#
# LANGCHAIN CONCEPT: ChatGoogleGenerativeAI
#   `langchain_google_genai.ChatGoogleGenerativeAI` wraps the Gemini API
#   behind LangChain's standard chat-model interface (`.invoke(messages)`),
#   the same interface every other LangChain-supported model uses. That
#   means the query rewriter (which also calls `.invoke(messages)`) and this
#   pipeline share one Gemini client instance and one calling convention --
#   swapping to a different LangChain-supported model later would not
#   require touching prompt_templates.py or query_rewriter.py at all.
#
# STREAMING
#   generate_answer()/answer_question() accept an optional `on_token`
#   callback. When given, the final answer streams token-by-token (via the
#   LangChain chat model's `.stream()` instead of `.invoke()`) and `on_token`
#   is called with the accumulated text so far on each new piece -- see
#   generate_answer()'s docstring. Retrieval, grounding, memory,
#   and MLflow logging are unaffected either way; only how the final answer
#   text is delivered changes.
#
# CONCURRENCY
#   Multi-query retrieval's per-variant searches (rag_pipeline/multi_query.py)
#   run concurrently via a thread pool in retrieve() -- they're
#   independent embedding + Chroma calls with no shared state, so running
#   them one after another was pure wasted wall-clock time. This is threads,
#   not asyncio: the underlying I/O (the embedding provider, ChromaDB) is
#   ordinary blocking Python, and threads already release the GIL during
#   network I/O, so a full async rewrite of the retrieval stack wouldn't buy
#   more concurrency here, just more code.
#
# INPUT / OUTPUT
#   answer_question(question) -> {"answer": str, "sources": [...],
#   "confidence": float} -- everything the Streamlit UI needs to render one
#   chat turn.
# ==============================================================================

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from rag_pipeline import guardrails, retrieval_service
from rag_pipeline.llm_service import acquire_chat_slot, get_chat_model
from rag_pipeline.memory import ConversationMemory
from rag_pipeline.multi_hop import plan_next_hop
from rag_pipeline.multi_query import generate_query_variants, reciprocal_rank_fusion
from rag_pipeline.prompt_templates import NOT_FOUND_MESSAGE, build_prompt
from rag_pipeline.query_rewriter import rewrite_query
from rag_pipeline.retrieval_service import RetrievedChunk, chunk_identity
from utils.logging_utils import get_logger
from utils.mlflow_tracking import trace_query

logger = get_logger(__name__)

SERVICE_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching the AI service right now (it may be rate-limited "
    "or temporarily unavailable). Please try asking again in a moment."
)
AUTH_ERROR_MESSAGE = (
    "I can't reach the AI service because its API key appears to be invalid, "
    "expired, or revoked. Please check GEMINI_API_KEY (or the active provider's "
    "credentials) in your .env file, then restart the app."
)

# Markers seen in real Gemini/Databricks auth failures (401 Unauthenticated,
# "invalid authentication credentials", a revoked/expired key). These are
# PERMANENT failures: retrying with backoff just wastes 10-20 seconds before
# failing again with the exact same error, and the generic "rate-limited or
# temporarily unavailable" message actively misleads the user into thinking
# the problem will resolve itself if they just wait -- it won't, until the
# key is fixed.
_PERMANENT_AUTH_ERROR_MARKERS = (
    "unauthenticated",
    "invalid authentication credentials",
    "permission_denied",
    "api key not valid",
    "api_key_invalid",
)


QUOTA_EXCEEDED_MESSAGE = (
    "The AI service's usage limit for this API key has been reached, so I can't "
    "generate an answer right now. If it's the per-minute limit, please wait about "
    "a minute and try again. If it keeps happening, the key's daily limit has been "
    "used up: try again after it resets, or upgrade the API key's plan for higher limits."
)

# Every message returned INSTEAD of an answer when the AI service itself
# failed. answer_question() uses this to return no sources and 0 confidence
# for such turns: showing retrieval scores under an error message suggests an
# answer was produced when none was.
SERVICE_ERROR_MESSAGES = frozenset(
    {SERVICE_UNAVAILABLE_MESSAGE, AUTH_ERROR_MESSAGE, QUOTA_EXCEEDED_MESSAGE}
)

# Gemini returns 429 RESOURCE_EXHAUSTED ("You exceeded your current quota")
# for both its per-minute and per-day free-tier limits.
_QUOTA_ERROR_MARKERS = (
    "resource_exhausted",
    "resourceexhausted",
    "exceeded your current quota",
    "quota exceeded",
    "rate limit",
    "429",
)
_QUOTA_ERROR_TYPES = ("ResourceExhausted", "TooManyRequests", "RateLimitError")


def _is_permanent_auth_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_AUTH_ERROR_MARKERS)


def _is_quota_error(exc: BaseException) -> bool:
    if type(exc).__name__ in _QUOTA_ERROR_TYPES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _QUOTA_ERROR_MARKERS)


def _is_daily_quota_error(exc: BaseException) -> bool:
    # e.g. quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
    # Unlike a per-minute limit, retrying within seconds can never succeed.
    return _is_quota_error(exc) and "perday" in str(exc).lower()


def _should_retry(exc: BaseException) -> bool:
    # Quota errors are never retried here. langchain-google-genai already
    # retries every 429 internally (a hard-coded 2 attempts), and Google
    # counts rejected requests against the same free-tier quotas (5/min,
    # 20/day). Retrying on top of that multiplied one rejected call into ~6
    # requests -- live, a single question exhausted the whole daily quota.
    # Timeouts (DeadlineExceeded) are likewise already retried once by that
    # library; retrying them here too would make one stalled request cost
    # up to ~6 x GEMINI_REQUEST_TIMEOUT_SECONDS before the user sees anything.
    return not (_is_permanent_auth_error(exc) or _is_quota_error(exc) or _is_timeout_error(exc))


def _is_timeout_error(exc: BaseException) -> bool:
    return type(exc).__name__ == "DeadlineExceeded" or "deadline exceeded" in str(exc).lower()


def _service_error_message(exc: BaseException) -> str:
    """The user-facing message for an exception raised by the AI service."""
    if _is_permanent_auth_error(exc):
        return AUTH_ERROR_MESSAGE
    if _is_quota_error(exc):
        return QUOTA_EXCEEDED_MESSAGE
    return SERVICE_UNAVAILABLE_MESSAGE


def _interleave(first: List[RetrievedChunk], second: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Alternate items from two best-first lists: first[0], second[0], first[1], ..."""
    merged: List[RetrievedChunk] = []
    for i in range(max(len(first), len(second))):
        merged.extend(lst[i] for lst in (first, second) if i < len(lst))
    return merged


def _normalize_answer(text: str) -> str:
    return text.strip().strip("\"'*_` ").rstrip(".").strip().lower()


def _is_not_found_answer(answer: str) -> bool:
    """
    True when the model's whole answer is the required "not found" sentence
    (tolerating quotes/markdown emphasis/a missing period around it). An
    answer that merely CONTAINS that sentence -- e.g. one part of a two-part
    question answered, the other not found -- is a real answer and keeps
    its sources.
    """
    return _normalize_answer(answer) == _normalize_answer(NOT_FOUND_MESSAGE)


class RAGPipeline:
    """
    Orchestrates one full question-answering turn: rewrite -> retrieve ->
    build context -> generate -> guardrail-check.

    One instance is created per Streamlit session (see frontend/app.py,
    cached via st.cache_resource) and reused across every question the user
    asks, so the underlying Gemini client and embedding model are only
    initialized once.
    """

    def __init__(self):
        settings.validate()
        self._llm = get_chat_model()
        self.memory = ConversationMemory()
        logger.info(
            "RAGPipeline ready (llm_provider=%s, embedding_provider=%s)",
            settings.llm_provider,
            settings.embedding_provider,
        )

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """
        Retrieve the top-K most relevant chunks for `question`.

        Runs vector (+ optional hybrid) search via retrieval_service --
        optionally against several LLM-generated paraphrasings of `question`
        (see rag_pipeline/multi_query.py), fused via reciprocal rank fusion
        when ENABLE_MULTI_QUERY is set -- and keeps the best top_k.

        The per-variant searches (when multi-query is on) are independent of
        each other -- each is its own embedding call + Chroma query with no
        shared state -- so they run concurrently via a thread pool instead of
        one after another. This is threads, not asyncio: the underlying work
        (the embedding provider, ChromaDB's client) is ordinary blocking I/O,
        not async-native, and threads release the GIL during that I/O just
        like async would, for a fraction of the rewrite. utils/rate_limiter.py
        (shared by the embedding provider) is already lock-protected, so
        concurrent callers pace correctly against the same quota rather than
        each under-counting the others' requests.

        The result can be shorter than top_k, or empty: retrieval_service
        already dropped chunks below the score threshold, and a fixed source
        count padded out with weak matches is exactly what a real relevance
        threshold is supposed to prevent.
        """
        if settings.enable_multi_query:
            variants = generate_query_variants(question, self._llm, settings.multi_query_variants)
            with ThreadPoolExecutor(max_workers=len(variants)) as executor:
                variant_results = list(executor.map(retrieval_service.retrieve, variants))
            candidates = reciprocal_rank_fusion(variant_results)
        else:
            candidates = retrieval_service.retrieve(question)
        return candidates[: settings.top_k]

    def _run_multi_hop(
        self, question: str, chunks: List[RetrievedChunk], context: str
    ) -> "tuple[List[RetrievedChunk], str]":
        """
        Let the model ask itself up to (settings.max_hops - 1) follow-up
        search queries when the current context doesn't fully answer
        `question` (see rag_pipeline/multi_hop.py), merging each hop's
        chunks into the running set and rebuilding the context each time.

        Bounded by settings.max_hops -- a single sufficiency-check failure
        or an already-sufficient context stops the loop immediately, so the
        common case costs nothing beyond the one extra check.
        """
        hops_done = 1
        seen_identities = {chunk_identity(c) for c in chunks}

        while hops_done < settings.max_hops:
            follow_up_query = plan_next_hop(question, context, self._llm)
            if not follow_up_query:
                break

            new_chunks = [c for c in self.retrieve(follow_up_query) if chunk_identity(c) not in seen_identities]
            if not new_chunks:
                # Nothing new came back -- another hop would just repeat
                # this same check against unchanged context.
                break

            # Cap at top_k like every other retrieval path -- simply
            # appending let a follow-up push the total to 2 x top_k (the UI
            # showed 9-10 "sources"). Interleaving (best of each round
            # first) instead of re-sorting by score guarantees the
            # follow-up's chunks survive the cap: they were fetched
            # precisely because the first round lacked that information.
            chunks = _interleave(chunks, new_chunks)[: settings.top_k]
            seen_identities.update(chunk_identity(c) for c in new_chunks)
            context = self.build_context(chunks)
            hops_done += 1

        return chunks, context

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into the citation-ready text block the LLM
        sees in its prompt.

        Each chunk is labeled with a bracketed source tag
        ([Source: <document>, page <N>], with chapter/section appended when
        the source document has that structure) directly above its text, so
        the model has an unambiguous, ready-to-copy citation for every piece
        of context it might use in its answer.
        """
        if not chunks:
            return "(No relevant context was found in the available documents.)"

        blocks = []
        for chunk in chunks:
            citation = f"{chunk.document_name}, page {chunk.page_number}"
            if chunk.section_title:
                citation += f" ({chunk.section_title})"
            blocks.append(f"[Source: {citation}]\n{chunk.chunk_text}")
        return "\n\n---\n\n".join(blocks)

    @retry(
        # Chat APIs (Gemini's free tier especially -- 5 requests/minute) can
        # return transient 429/503 errors under normal interactive use, not
        # just under load. A short, bounded retry smooths over a single
        # rate-limit blip without making the user wait too long if the
        # service is genuinely down.
        #
        # retry_error_callback below re-raises immediately (no backoff at
        # all) when the underlying error is a permanent auth failure (a
        # dead/invalid API key): retrying that with backoff just delays the
        # same guaranteed failure by 10-20 seconds for nothing.
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=lambda retry_state: (
            retry_state.outcome.failed and _should_retry(retry_state.outcome.exception())
        ),
        reraise=True,
    )
    def _invoke_llm(self, messages: List) -> str:
        acquire_chat_slot()
        response = self._llm.invoke(messages)
        return response.content.strip()

    @retry(
        # Same policy as _invoke_llm() above -- this decorator wraps ONE
        # attempt at streaming the full response. If a failure happens after
        # some chunks were already sent to `on_token`, the retried attempt
        # starts its own accumulation from empty and calls `on_token` with
        # that fresh (shorter) text -- the caller's UI naturally re-renders
        # from scratch rather than appending a stale partial answer to a new
        # one, since each call passes the full accumulated-so-far text, not
        # a delta.
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=lambda retry_state: (
            retry_state.outcome.failed and _should_retry(retry_state.outcome.exception())
        ),
        reraise=True,
    )
    def _stream_llm(self, messages: List, on_token: Callable[[str], None]) -> str:
        acquire_chat_slot()
        accumulated = ""
        for chunk in self._llm.stream(messages):
            piece = chunk.content or ""
            if piece:
                accumulated += piece
                on_token(accumulated)
        return accumulated.strip()

    def generate_answer(
        self, context: str, question: str, on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Call the chat model with the grounded prompt and return its raw text
        answer.

        If `on_token` is given, the answer streams: `on_token` is called
        with the accumulated text so far every time a new piece arrives
        (e.g. for a Streamlit placeholder to re-render), and this method
        still returns the final complete string once streaming finishes --
        everything downstream (grounding check, memory, MLflow logging)
        works from that same final string exactly as it does today, so
        streaming is purely a display-layer change, not a different answer.
        Without `on_token`, this makes one plain, non-streaming call.

        If the LLM call still fails (e.g. a sustained rate limit, an outage,
        or an invalid/revoked API key), this returns a clear, user-facing
        message instead of letting an unhandled exception reach the
        Streamlit UI as a raw traceback -- a degraded but honest response
        beats a crashed chat turn. A permanent auth failure gets its own
        distinct message rather than the generic "try again in a moment"
        one, since waiting will never fix a dead key. An exhausted usage
        quota (429) likewise gets its own message naming the key's limit.
        """
        messages = build_prompt(context, question, self.memory.get_history())
        try:
            if on_token is not None:
                return self._stream_llm(messages, on_token)
            return self._invoke_llm(messages)
        except Exception as exc:
            logger.exception("Chat model call failed")
            return _service_error_message(exc)

    def answer_question(
        self, question: str, on_token: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Run one full chat turn end-to-end: input guardrail -> query rewrite
        -> retrieve -> build context -> generate -> grounding
        check -> update memory.

        `on_token`, if given, is forwarded to generate_answer() so the final
        answer streams instead of arriving all at once -- see that method's
        docstring. It has no effect on the guardrail-rejection or "not
        found" fallback paths, which return instantly with no LLM call
        either way.

        Returns:
            {
              "answer": str,
              "sources": [{"document_name", "document_type", "page_number", "score"}, ...],
              "confidence": float,  # average retrieval score of chunks actually used, 0 if none
              "grounded": bool,     # heuristic grounding check result
              "service_error": bool,  # True if the AI service failed and "answer" is an
                                      # error message (then sources=[] and confidence=0)
            }
        """
        is_allowed, rejection_reason = guardrails.check_input(question)
        if not is_allowed:
            return {
                "answer": rejection_reason,
                "sources": [],
                "confidence": 0.0,
                "grounded": True,
                "service_error": False,
            }

        active_chat_model = (
            settings.gemini_chat_model
            if settings.llm_provider == "gemini"
            else settings.databricks_llm_endpoint
        )
        with trace_query(question, top_k=settings.top_k, chat_model=active_chat_model) as run_data:
            standalone_question = rewrite_query(question, self.memory.get_history(), self._llm)

            try:
                chunks = self.retrieve(standalone_question)
            except Exception as exc:
                # The embedding call inside retrieve() has no fallback of
                # its own (unlike generate_answer() below) -- an invalid
                # API key or a dead embedding endpoint would otherwise
                # crash this whole method with a raw traceback reaching the
                # Streamlit UI. Degrade the same way generate_answer() does.
                logger.exception("Retrieval failed")
                answer = _service_error_message(exc)
                run_data["answer"] = answer
                run_data["retrieved_chunks"] = []
                return {
                    "answer": answer,
                    "sources": [],
                    "confidence": 0.0,
                    "grounded": True,
                    "service_error": True,
                }

            if not chunks:
                # No relevant context at all -- return the required fallback
                # sentence directly, without ever calling the LLM. This
                # guarantees the exact required wording and avoids the
                # (small but real) risk of the model answering from its own
                # general knowledge when given empty context.
                answer = NOT_FOUND_MESSAGE
                sources: List[Dict[str, Any]] = []
                confidence = 0.0
                is_grounded = True
            else:
                context = self.build_context(chunks)
                if settings.enable_multi_hop:
                    chunks, context = self._run_multi_hop(standalone_question, chunks, context)
                answer = self.generate_answer(context, standalone_question, on_token=on_token)
                if answer in SERVICE_ERROR_MESSAGES:
                    # No answer was generated, so no sources or confidence:
                    # retrieval scores shown under an error message read as
                    # if the bot had answered from those pages.
                    run_data["answer"] = answer
                    run_data["retrieved_chunks"] = []
                    return {
                        "answer": answer,
                        "sources": [],
                        "confidence": 0.0,
                        "grounded": True,
                        "service_error": True,
                    }
                if _is_not_found_answer(answer):
                    # The model read the retrieved chunks and found no
                    # answer in them. Listing those chunks as "sources", with
                    # their retrieval scores as "confidence", would present
                    # a non-answer as if it were backed by 5 documents.
                    answer = NOT_FOUND_MESSAGE
                    sources = []
                    confidence = 0.0
                    is_grounded = True
                else:
                    sources = [
                        {
                            "document_name": c.document_name,
                            "document_type": c.document_type,
                            "page_number": c.page_number,
                            "score": c.score,
                            "chapter_title": c.chapter_title,
                            "section_title": c.section_title,
                        }
                        for c in chunks
                    ]
                    confidence = sum(c.score for c in chunks) / len(chunks)
                    is_grounded = guardrails.check_grounding(answer, [c.chunk_text for c in chunks])

            # Only real answers go into conversation memory -- an error
            # message in the history would be fed to the query rewriter and
            # the prompt on the next turn.
            self.memory.add_turn(question, answer)

            run_data["answer"] = answer
            run_data["retrieved_chunks"] = sources

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3),
            "grounded": is_grounded,
            "service_error": False,
        }
