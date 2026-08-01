FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
RUN pip install --no-cache-dir --no-deps -e .

RUN mkdir -p data/raw checkpoints

ENTRYPOINT ["graph-ml"]
CMD ["--help"]
