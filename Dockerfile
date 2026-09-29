# ==============================================================================
# Dockerfile -- container image for running the chatbot on any Docker host
# (a company VM/sandbox, a cloud container service, ...).
#
#   docker build -t insurance-chatbot .
#   docker run -p 8501:8501 -e GEMINI_API_KEY=... insurance-chatbot
#
# No model is downloaded (embeddings and answers both come from the Gemini
# API), so the image stays small. The vector index needs GEMINI_API_KEY to
# embed the PDFs, so it's built when the container STARTS rather than at
# build time: `scripts.ingest` indexes data/pdfs on first start and skips
# unchanged files after that (mount a volume on vectorstore_db/ to keep the
# index -- and feedback.jsonl -- across container restarts).
#
# Other settings come from environment variables (`-e NAME=value`), exactly
# like the local .env file -- .env itself is never copied into the image
# (.dockerignore).
# ==============================================================================

FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
CMD ["sh", "-c", "python -m scripts.ingest && exec streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
