FROM --platform=linux/amd64 ghcr.io/shivansh-bajaj/amd-hackathon@sha256:a906d510c4ffa8b2e7d5de66a7840b34f6652af51158147f82b690179218920b AS local_llm

FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_PATH=/models/gemma-4-E2B-it-Q4_K_M.gguf
ENV ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"
ENV CHEAPEST_MODEL="minimax-m3"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=local_llm /usr/local/bin/llama-server /usr/local/bin/llama-server
COPY --from=local_llm /models /models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "import spacy; spacy.load('en_core_web_sm')"

COPY . .
RUN chmod +x run.sh

ENTRYPOINT ["bash", "run.sh"]
