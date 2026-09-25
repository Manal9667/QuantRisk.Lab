# QuantumRiskLab API image.
# Python 3.11 is pinned deliberately: Qiskit and the scientific stack have
# mature, well-tested wheels there.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime system deps: libpq for psycopg, curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first for better layer caching.
COPY requirements.txt requirements-quantum.txt ./
RUN pip install -r requirements.txt -r requirements-quantum.txt

# Project sources.
COPY pyproject.toml README.md ./
COPY src ./src
COPY sql ./sql
COPY migrations ./migrations
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY docker ./docker

# Normalise line endings (a Windows checkout may carry CRLF, which breaks the
# shell shebang inside the Linux container) and make the entrypoint executable.
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh

# Install the package (console scripts + importable path resolution).
RUN pip install -e .

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["api"]
