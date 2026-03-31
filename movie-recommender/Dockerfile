# ============================================================
# Dockerfile — production image for the FastAPI service
# Python 3.12-slim (matches local dev environment)
#
# uvloop + httptools installed here (Linux only — not in
# requirements.txt which is shared with Windows local dev)
# ============================================================

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install base requirements + Linux-only performance deps
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install \
        uvloop==0.19.0 \
        httptools==0.6.1


FROM python:3.12-slim AS final

WORKDIR /app

COPY --from=builder /install /usr/local

# FAISS SIMD fix for containerized environments
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=info
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

COPY src/ ./src/
COPY configs/ ./configs/
COPY train.py evaluate.py ./

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]