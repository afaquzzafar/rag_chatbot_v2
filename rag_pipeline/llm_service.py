# ==============================================================================
# rag_pipeline/llm_service.py
# ------------------------------------------------------------------------------
# STEP: chooses which chat LLM answers questions -- Gemini (needs an API
# key) or a Databricks-hosted Foundation Model (fully keyless inside
# Databricks). Mirrors the exact factory pattern used in
# embeddings/embedding_service.py, for the same reason: the rest of the app
# (rag_pipeline.py, query_rewriter.py) should depend only on "a LangChain
# chat model with .invoke(messages)", never on which vendor is behind it.
#
# WHY "FULLY KEYLESS" IS A REAL, MEANINGFUL DEPLOYMENT MODE
#   Every LLM API (Gemini included) normally requires your application to
#   hold a secret credential -- something that can leak, expire, or need
#   rotating. Databricks Foundation Model APIs are different when your code
#   *runs inside the same Databricks workspace* that hosts the model serving
#   endpoint: `langchain_databricks.ChatDatabricks` authenticates via the
#   Databricks SDK's ambient credential resolution, which inside a
#   Databricks notebook, job, or App requires zero configuration -- there is
#   no API key anywhere in this app's environment, config files, or code.
#   Running this same code from outside Databricks (e.g. testing from your
#   laptop against a real workspace) still works, using DATABRICKS_HOST +
#   DATABRICKS_TOKEN as an explicit fallback credential.
#
# PACING CHAT CALLS AGAINST GEMINI'S STRICT FREE-TIER QUOTA
#   Live testing showed Gemini's free tier caps CHAT generation at just 5
#   requests/minute -- far stricter than the ~100/minute embedding quota,
#   and easy to exhaust because both the query-rewrite call
#   (query_rewriter.py) and the main answer-generation call
#   (rag_pipeline.py) draw from this SAME quota. `acquire_chat_slot()` is
#   the one place both call sites pace themselves through, sharing a single
#   rate limiter so the app sees the true combined request rate rather than
#   each call site tracking (and under-counting) its own.
#
# INPUT / OUTPUT
#   get_chat_model() -> a LangChain BaseChatModel (ChatGoogleGenerativeAI or
#   ChatDatabricks), built once per process and reused.
#   acquire_chat_slot() -> blocks (sleeping if needed) until it's safe to
#   make one more chat call without exceeding the active provider's quota.
# ==============================================================================

from typing import Optional

from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_llm_instance = None
_chat_rate_limiter = None


def get_chat_model():
    """
    Return the process-wide chat model singleton, built on first call based
    on `settings.llm_provider`.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if settings.llm_provider == "databricks":
        from langchain_databricks import ChatDatabricks

        logger.info(
            "Using Databricks-hosted chat model (endpoint='%s') -- keyless "
            "when run inside a Databricks workspace",
            settings.databricks_llm_endpoint,
        )
        _llm_instance = ChatDatabricks(
            endpoint=settings.databricks_llm_endpoint,
            temperature=settings.databricks_temperature,
            # host/token are only passed when explicitly set; leaving them
            # None lets the Databricks SDK fall back to the ambient
            # workspace identity when running inside Databricks itself.
            host=settings.databricks_host or None,
            token=settings.databricks_token or None,
        )
    elif settings.llm_provider == "gemini":
        logger.info("Using Gemini chat model (model='%s')", settings.gemini_chat_model)
        _llm_instance = _gemini_chat_model_class()(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature,
            # Without a timeout a stalled connection blocks the chat turn
            # forever (seen live: "Searching your plan documents..." for 4+
            # minutes while the same request succeeds in ~8s on retry).
            timeout=settings.gemini_request_timeout_seconds,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{settings.llm_provider}'")

    return _llm_instance


def _gemini_chat_model_class():
    """
    ChatGoogleGenerativeAI, but with its `timeout` actually applied.

    langchain-google-genai 2.0.x accepts `timeout` on the chat class but never
    forwards it to the Gemini client's generate_content /
    stream_generate_content calls (only its plain-text LLM class does), so
    without this a request can hang indefinitely. Extra keyword arguments to
    _generate/_stream ARE forwarded to those client calls, which accept
    `timeout=` (a gRPC deadline, covering a whole streamed response too).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    class TimeoutChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.timeout:
                kwargs.setdefault("timeout", self.timeout)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            if self.timeout:
                kwargs.setdefault("timeout", self.timeout)
            return super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    return TimeoutChatGoogleGenerativeAI


def acquire_chat_slot() -> None:
    """
    Block until it's safe to make one more chat call without exceeding the
    active provider's known rate limit. Call this immediately before every
    `llm.invoke(...)` that reaches the chat model -- both the query
    rewriter's call and the main answer-generation call.

    Databricks Foundation Model APIs don't have a comparably strict,
    universally-documented free-tier cap the way Gemini's free tier does,
    so this is a no-op for that provider; add pacing here if your workspace
    endpoint has a known throughput limit worth respecting.
    """
    global _chat_rate_limiter
    if settings.llm_provider != "gemini":
        return

    if _chat_rate_limiter is None:
        from utils.rate_limiter import SlidingWindowRateLimiter

        _chat_rate_limiter = SlidingWindowRateLimiter(
            settings.gemini_chat_requests_per_minute, name="Gemini chat requests"
        )
    _chat_rate_limiter.acquire(1)
