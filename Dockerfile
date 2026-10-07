FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models

WORKDIR /app

# Tesseract OCR + language packs (space separated Debian suffixes, e.g. "eng hin ben")
ARG OCR_LANG_PACKS="eng hin"
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
       $(for l in $OCR_LANG_PACKS; do echo tesseract-ocr-$l; done) \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the HuggingFace models into the image (cached until the model names change)
ARG EMBEDDING_PROVIDER=huggingface
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ARG RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
COPY scripts/download_models.py scripts/download_models.py
RUN EMBEDDING_PROVIDER=$EMBEDDING_PROVIDER EMBEDDING_MODEL=$EMBEDDING_MODEL RERANKER_MODEL=$RERANKER_MODEL \
    python scripts/download_models.py

COPY src ./src
COPY ui ./ui
COPY scripts ./scripts

EXPOSE 8000 8501

CMD ["uvicorn", "graphrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
