FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg (binary) + git (for commit.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Optimizer-specific requirements (lighter than full pipeline)
COPY requirements-optimizer.txt ./
RUN pip install --no-cache-dir -r requirements-optimizer.txt

# Copy project source
COPY . .

# Entry point: python -m optimizer <command>
ENTRYPOINT ["python", "-m", "optimizer"]
CMD ["--help"]
