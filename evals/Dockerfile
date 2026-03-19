FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Eval harness HTTP server — implemented in Foundation Step 5
# For now, keeps the container alive so docker-compose doesn't restart-loop it
CMD ["python", "-c", "import time; print('eval-harness: not yet implemented (Step 5)'); time.sleep(3600)"]
