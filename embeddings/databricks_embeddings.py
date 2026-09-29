# ==============================================================================
# embeddings/databricks_embeddings.py
# ------------------------------------------------------------------------------
# FULLY-KEYLESS PROVIDER (when this app itself runs inside Databricks).
#
# WHAT THIS FILE DOES
#   Implements EmbeddingProvider using a Databricks Model Serving endpoint
#   -- typically one of Databricks' pay-per-token Foundation Model APIs
#   (e.g. "databricks-gte-large-en") -- via the `databricks-sdk`.
#
# WHY THIS IS "FULLY KEYLESS"
#   The Gemini provider needs a credential your code has to hold (a
#   GEMINI_API_KEY). This provider authenticates through
#   `databricks.sdk.WorkspaceClient()`, which uses the Databricks SDK's
#   standard auth resolution chain: explicit DATABRICKS_HOST/DATABRICKS_TOKEN
#   if set (useful for testing from outside a workspace), falling back to
#   `~/.databrickscfg`, and -- the important case -- automatically picking
#   up the ambient workspace identity with ZERO configuration when this code
#   runs inside a Databricks notebook, job, or Databricks App. In that
#   deployment mode, no API key of any kind appears anywhere in this app's
#   config, environment, or code.
#
# WHY A SEPARATE FILE (same reasoning as gemini_embeddings.py)
#   Importing this module requires `databricks-sdk` to be installed, but a
#   developer using EMBEDDING_PROVIDER=gemini should never need
#   that dependency installed at all -- the lazy `import` inside __init__
#   keeps it optional.
#
# INPUT / OUTPUT
#   Input:  list of strings (documents) or one string (query).
#   Output: list of float vectors (or one vector for embed_query).
# ==============================================================================

from typing import Any, Dict, List

from config.settings import settings
from embeddings.base import EmbeddingProvider
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabricksEmbeddingProvider(EmbeddingProvider):
    """Embeddings via a Databricks Model Serving endpoint -- no app-level API key."""

    def __init__(self, endpoint_name: str):
        from databricks.sdk import WorkspaceClient

        # WorkspaceClient() with no arguments resolves credentials in this
        # order: explicit host/token kwargs (not used here) -> DATABRICKS_HOST
        # /DATABRICKS_TOKEN env vars -> ~/.databrickscfg -> the ambient
        # workspace identity when running inside Databricks itself.
        self._client = WorkspaceClient(
            host=settings.databricks_host or None,
            token=settings.databricks_token or None,
        )
        self._endpoint_name = endpoint_name
        logger.info("Databricks embedding provider ready (endpoint='%s')", endpoint_name)

    def _query_endpoint(self, texts: List[str]) -> List[List[float]]:
        response = self._client.serving_endpoints.query(
            name=self._endpoint_name,
            input=texts,
        )
        # The Foundation Model APIs embedding response follows the OpenAI-style
        # shape: response.data is a list of {"embedding": [...], "index": i}.
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = settings.embedding_batch_size
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            all_vectors.extend(self._query_endpoint(batch))
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        return self._query_endpoint([text])[0]
