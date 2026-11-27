FROM pgvector/pgvector:0.8.1-pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends 'postgresql-16-age=1.6.0~rc0-2.pgdg12+2' \
    && rm -rf /var/lib/apt/lists/*
