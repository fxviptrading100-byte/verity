FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
gcc \
g++ \
libpq-dev \
curl \
&& rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir \
--extra-index-url https://download.pytorch.org/whl/cpu \
-r requirements.txt
COPY env/ ./env/
COPY model/ ./model/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY inference.py .
COPY README.md .
RUN mkdir -p data
ENV PORT=8001
ENV PYTHONPATH=/app
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
CMD curl -f http://localhost:${PORT}/health || exit 1
CMD uvicorn env.verity_openenv:app --host 0.0.0.0 --port ${PORT}
