FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Only need minimal dependencies
RUN pip install --no-cache-dir click rich requests

COPY cli_help/ ./cli_help/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m", "cli_help.main"]