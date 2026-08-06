FROM python:3.12-slim

# Pass these at runtime: docker run -e API_BASE_URL=... -e API_KEY=...
ARG API_BASE_URL
ARG API_KEY
ENV API_BASE_URL=${API_BASE_URL}
ENV API_KEY=${API_KEY}

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8050

CMD ["uv", "run", "gunicorn", "app:server", "-b", "0.0.0.0:8050", "--workers", "2", "--timeout", "120"]
