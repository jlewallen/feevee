FROM python:3.9-slim AS python-venv-image

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc
RUN python -m venv /app/venv
# Make sure we use the virtualenv:
ENV PATH="/app/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM node:16 AS build-nodejs-image

WORKDIR /app
COPY package.json /app/web/
RUN cd web && yarn install 
COPY public /app/web/public
COPY src /app/web/src
COPY src/config.ts.prod /app/src/web/src/config.ts
RUN cd web && yarn build

FROM python:3.9-slim

WORKDIR /app

ENV PATH="/app/venv/bin:$PATH"

COPY --from=python-venv-image /app/venv /app/venv
COPY --from=build-nodejs-image /app/web/dist /app/dist/
COPY *.json /app/
COPY *.py /app/
COPY logging.json /app/

# TODO Sorry
ENV FEEVEE_REDIS_HOST 172.17.0.2
ENV TZ America/Los_Angeles

ENTRYPOINT [ "uvicorn", "--log-config", "/app/logging.json", "--host", "0.0.0.0", "--port", "8000", "--factory", "app:factory" ]
