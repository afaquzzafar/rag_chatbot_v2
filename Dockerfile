# ==============================================================================
# Dockerfile -- container image for hosting the chatbot (e.g. a Hugging Face
# Docker Space; see the YAML header at the top of README.md).
#
# Everything slow or network-dependent happens at BUILD time, so the running
# app starts quickly and needs only GEMINI_API_KEY at runtime:
#   - dependencies (CPU-only torch, per requirements.txt)
#   - the local embedding model and the reranker model (Hugging Face Hub)
#   - the vector index over data/pdfs (local embeddings, no API key needed)
#
# Runtime configuration comes from environment variables (Space "Variables and
# secrets"), exactly like the local .env file -- .env itself is never copied
# into the image (.dockerignore).
# ==============================================================================

FROM python:3.12-slim

# Hugging Face Spaces run containers as uid 1000; the app writes its vector
# store, MLflow runs and feedback log under the app directory at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Bake the models and the index into the image. run_ingestion() is called
# directly (not `python -m scripts.ingest`) because the CLI entry point
# validates GEMINI_API_KEY, which isn't available -- or needed -- at build time.
RUN python -c "from sentence_transformers import CrossEncoder; from config.settings import settings; CrossEncoder(settings.reranker_model)" \
 && python -c "from scripts.ingest import run_ingestion; print(run_ingestion())"

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
