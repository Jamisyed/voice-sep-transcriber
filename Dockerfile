# Use slim Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code into /app
COPY src ./src
COPY README.md .

# Expose API port
EXPOSE 8000

# Env vars
ENV PORT=8000
ENV NUM_WORKERS=1
ENV PYTHONPATH=/app   

# Run FastAPI app
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
