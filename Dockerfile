FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/
COPY README.md .

RUN pip install --no-cache-dir -e ".[dev]"

# Run tests during build to validate
RUN pytest tests/ -v --tb=short

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/viz/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
