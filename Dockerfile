# syntax=docker/dockerfile:1
FROM python:3.13-slim

RUN pip install --no-cache-dir uv \
    && useradd --create-home --uid 1000 wordvec \
    && mkdir -p /app /data \
    && chown wordvec:wordvec /app /data

# Everything below runs as the unprivileged user, so nothing needs a recursive
# chown afterwards (which would copy every file into a second layer).
USER wordvec
WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    WORDVEC_CACHE_DIR=/data/cache \
    GENSIM_DATA_DIR=/data/gensim-data

# Dependencies first (separate layer) so code-only changes don't rebuild it.
COPY --chown=wordvec:wordvec pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake the models into the image (English GloVe ~65 MB, Chinese vectors
# ~116 MB, CC-CEDICT ~4 MB) so the container never needs the internet at run
# time. datafiles.py has no heavy imports, so this layer is cached until the
# model URLs change -- editing the code does not re-download anything.
COPY --chown=wordvec:wordvec src/wordvec/datafiles.py /tmp/datafiles.py
RUN python /tmp/datafiles.py && rm /tmp/datafiles.py

COPY --chown=wordvec:wordvec src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000

# --workers 1 --preload: the app keeps ~400 MB of vectors in memory and runs
# one command at a time under a lock, so a second worker would only double the
# RAM. --preload loads both models before the port opens, which is what lets
# idle-gate's "is it reachable yet?" probe mean "ready to serve".
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--preload", "wordvec.web:build_app()"]
