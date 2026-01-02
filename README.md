# Translation Overlay Extension Server

A FastAPI proxy server for LLM translation requests with Redis caching.

## Features

- **OpenAI-compatible API**: Drop-in replacement endpoint `/v1/chat/completions`
- **Authentication**: `site_auth` header validation
- **Dynamic routing**: `site_api` header for custom LLM endpoints
- **Redis caching**: 3-day TTL for translation results
- **Auto-logging**: Daily JSON logs with auto-compression at 300MB

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Redis connection string

# Run server
uvicorn main:app --reload --port 8000
```

## API Usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "site_auth: YXBpLTEyMzQ1Ng==" \
  -H "site_api: https://api.openai.com/v1/chat/completions" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

## Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token for LLM API |
| `site_auth` | Yes | Fixed token `YXBpLTEyMzQ1Ng==` (401 if invalid) |
| `site_api` | No | Target LLM endpoint (default: llmproai.xyz) |

## Cache Key

Cache key is generated from `hash(messages array)`, ignoring temperature and other params.
