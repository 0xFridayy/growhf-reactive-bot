FROM python:3.12-slim

# Unbuffered so `docker logs` shows output live instead of in 4KB bursts, and
# no .pyc litter in the layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

WORKDIR /app

# Dependencies first — this layer is cached until requirements.txt changes,
# so code edits rebuild in seconds rather than re-installing matplotlib.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source. .dockerignore keeps config.json (and the venv, .git,
# runtime artifacts) out — config.json is mounted at runtime instead, so the
# Telegram token is never baked into an image layer.
COPY . .

# Drop privileges. Writable state lives in mounted volumes, not the image.
RUN useradd --create-home --uid 10001 botuser \
    && mkdir -p /app/logs /app/alpha_ml \
    && chown -R botuser:botuser /app
USER botuser

# Default service; docker-compose overrides this per container.
CMD ["python", "okx_tele_bot.py"]
