# syntax=docker/dockerfile:1
# TORCH_VARIANT=cpu  -> slim inference image (default)
# TORCH_VARIANT=cu130 -> CUDA training image (needs `--gpus all` / compose GPU reservation)
ARG TORCH_VARIANT=cpu

# ---------------------------------------------------------------- builder
FROM python:3.13-slim AS builder
ARG TORCH_VARIANT
ARG SLIM=true
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN pip install --prefix=/install torch==2.14.1 \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
        --extra-index-url https://pypi.org/simple
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install --no-deps .

# Slimming: bundled test suites, bytecode caches, type stubs and C/C++ headers are not
# needed to run the model.
RUN if [ "$SLIM" = "true" ]; then \
        find /install -depth -type d \( -name tests -o -name __pycache__ -o -name include \) \
            -exec rm -rf {} + ; \
        find /install \( -name "*.pyi" -o -name "*.a" \) -delete ; \
    fi

# ---------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime
ARG SLIM=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CIFAR_API_HOST=0.0.0.0 \
    CIFAR_API_PORT=8000

RUN if [ "$SLIM" = "true" ]; then \
        pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true ; \
        rm -rf /usr/local/lib/python3.13/ensurepip ; \
    fi \
    && useradd --create-home --uid 1000 app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app config ./config
COPY --chown=app:app data/manifest.json ./data/manifest.json
RUN mkdir -p data/raw models reports && chown -R app:app data models reports

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["cifar", "serve"]
