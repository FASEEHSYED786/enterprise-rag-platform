FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[pdf]" && useradd --system --uid 10001 rag && mkdir /data && chown rag /data
USER rag
ENV RAG_DB_PATH=/data/ragplatform.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "ragplatform.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
