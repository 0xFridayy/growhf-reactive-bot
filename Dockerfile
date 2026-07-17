FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY growhf_reactive_bot.py .
COPY okx_perp_screener.py .
COPY config.json .

# Run bot
CMD ["python", "growhf_reactive_bot.py"]
