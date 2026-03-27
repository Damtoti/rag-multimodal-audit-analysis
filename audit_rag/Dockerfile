FROM python:3.12-slim
 
RUN apt-get update && apt-get install -y \\
    ghostscript \\
    poppler-utils \\
    libglib2.0-0 \\
    libsm6 \\
    libxrender1 \\
    libxext6 \\
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && poetry install --no-dev --no-root
 
COPY src/ ./src/
COPY scripts/ ./scripts/
 
RUN poetry install --no-dev
 
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "audit_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]