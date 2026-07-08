FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "import spacy; spacy.load('en_core_web_sm')"

COPY . .
RUN chmod +x run.sh

ENTRYPOINT ["bash", "run.sh"]
