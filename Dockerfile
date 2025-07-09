FROM python:3.12.10-slim

RUN apt-get update && apt-get install -y curl

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml .
COPY uv.lock .
RUN uv venv
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

COPY . .

ENTRYPOINT [ "uv", "run", "python" ]
