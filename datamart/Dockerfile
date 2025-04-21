FROM postgres:16

RUN apt-get update && apt-get install -y \
    postgresql-16-cron \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY data.csv /docker-entrypoint-initdb.d/
COPY init.sql /docker-entrypoint-initdb.d/

RUN chmod a+r /docker-entrypoint-initdb.d/*