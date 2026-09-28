# ==============================================================================
# tests/test_provider_switching.py
# ------------------------------------------------------------------------------
# Tests that config/settings.py's validate() correctly enforces (or does NOT
# enforce) credentials depending on LLM_PROVIDER / EMBEDDING_PROVIDER --
# covering the "Gemini can be fully replaced by keyless Databricks" claim.
#
# These are pure config/logic tests -- no real Databricks workspace or
# Gemini key is used or needed; we never actually construct a chat model or
# call an endpoint here, only exercise the validation rules.
# ==============================================================================

import pytest

from config.settings import settings


def test_validate_requires_gemini_key_when_llm_provider_is_gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "gemini_api_key", "")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.validate()


def test_validate_passes_fully_keyless_when_both_providers_are_databricks(monkeypatch):
    # This is the "fully replace Gemini" configuration: no Gemini key
    # anywhere, both chat and embeddings routed to Databricks.
    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(settings, "embedding_provider", "databricks")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "databricks_host", "")
    monkeypatch.setattr(settings, "databricks_token", "")

    # Should not raise -- Databricks auth is resolved at call time by the
    # SDK's own ambient-credential chain, never validated as a config value.
    settings.validate()


def test_validate_rejects_unknown_llm_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        settings.validate()


def test_validate_rejects_unknown_embedding_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(settings, "embedding_provider", "azure_openai")

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        settings.validate()


def test_get_chat_model_dispatches_to_gemini(monkeypatch):
    import rag_pipeline.llm_service as llm_service

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    model = llm_service.get_chat_model()

    from langchain_google_genai import ChatGoogleGenerativeAI

    assert isinstance(model, ChatGoogleGenerativeAI)


@pytest.mark.parametrize("call", ["invoke", "stream"])
def test_gemini_chat_model_forwards_request_timeout_to_the_client(monkeypatch, call):
    # langchain-google-genai 2.0.x accepts `timeout` on the chat class but
    # never passes it to the Gemini client, so a stalled request hung the
    # chat turn forever. The app's subclass must forward it on both paths.
    import langchain_google_genai.chat_models as lc_chat_models
    import rag_pipeline.llm_service as llm_service

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "gemini_request_timeout_seconds", 42.0)
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    captured = {}

    class StopHere(Exception):
        pass

    def fake_chat_with_retry(generation_method, **kwargs):
        captured.update(kwargs)
        raise StopHere

    monkeypatch.setattr(lc_chat_models, "_chat_with_retry", fake_chat_with_retry)

    model = llm_service.get_chat_model()
    with pytest.raises(StopHere):
        if call == "invoke":
            model.invoke("hello")
        else:
            list(model.stream("hello"))

    assert captured["timeout"] == 42.0


def test_get_chat_model_dispatches_to_databricks(monkeypatch):
    import rag_pipeline.llm_service as llm_service

    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    model = llm_service.get_chat_model()

    assert type(model).__name__ == "ChatDatabricks"
