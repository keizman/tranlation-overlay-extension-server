# Translation Overlay Extension Server

FastAPI server with:
1. Core OpenAI-compatible proxy (`/v1/chat/completions`)
2. Enladder-compatible word extraction (`/api/app/v1/words/extract`)

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

## Auth Model (Code Truth)

- Outer auth token (client -> server):
  - `Authorization: Bearer <SITE_AUTH_TOKEN>`
  - Default `SITE_AUTH_TOKEN=YXBpLTEyMzQ1Ng==`
- Upstream LLM auth for `/v1/chat/completions`, priority:
  1. `LLM_SITE_AUTH` (env)
  2. `site_auth` header
  3. outer auth token
- Upstream LLM endpoint:
  - `site_api` header if provided, else `DEFAULT_LLM_ENDPOINT`

## API List

### Core Interface

1. `POST /v1/chat/completions`
   - OpenAI-compatible passthrough + Redis cache
   - Includes cache key normalization (`normalize_for_cache`)
   - Existing/original behavior is preserved

2. `GET /health`
3. `GET /config/cache-ttl`
4. `POST /config/cache-ttl`
5. `GET /cache/stats`

### Enladder-Compatible Interface

1. `POST /api/app/v1/words/extract`
   - Requires outer auth:
     - `Authorization: Bearer <SITE_AUTH_TOKEN>`
   - Request fields:
     - `text`, `annotationMode`, `filterMode`, `frequency`, `outputMode`, `lang`, `targetLanguage`, `maxWords`, `model`
   - Response:
     - Backward-compatible field:
       - `data.translations` (`{ "word": "译文" }`)
     - Additional fields:
       - `requestId`
       - `data.items` (`[{ "original", "translation", "zipf", "rank" }]`)
       - `data.meta` (`engine`, `detectedSourceLanguage`, `targetLanguage`, `filterMode`, `frequency`, `latencyMs`, ...)
   - Engine:
     - `wordfreq + dictionary-batch` (non-LLM)
   - Translation source:
     - Calls dictionary server batch API (`POST /api/v2/translate/batch`)
   - Supported pairs:
     - `en -> zh-CN`
     - `en -> ja|ko|de|ru`
     - `ja|ko|de|ru -> en`
   - Unsupported pairs:
     - returns `422` with code `UNSUPPORTED_LANGUAGE_PAIR`

## Cache Behavior

- `normalize_for_cache` is only used by:
  - `POST /v1/chat/completions`
- `/api/app/v1/words/extract` does not use cache normalization.

## Environment Variables

```env
SITE_AUTH_TOKEN=YXBpLTEyMzQ1Ng==
LLM_SITE_AUTH=
DEFAULT_LLM_ENDPOINT=http://127.0.0.1:8317/v1/chat/completions
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite
FORCE_DEFAULT_LLM_MODEL=true
DEFAULT_COMPAT_TARGET_LANGUAGE=zh-CN
DICTIONARY_SERVER_BASE_URL=http://127.0.0.1:9000
DICTIONARY_SERVER_BATCH_PATH=/api/v2/translate/batch
DICTIONARY_SERVER_TIMEOUT_MS=20000
REDIS_CONN_STRING=redis://localhost:6379/0
DEFAULT_CACHE_TTL_DAYS=30
LOG_DIR=logs
MAX_LOG_SIZE_MB=300
```

`DEFAULT_LLM_MODEL` and `FORCE_DEFAULT_LLM_MODEL` control upstream model selection:
- `FORCE_DEFAULT_LLM_MODEL=true` (default): always override request `model` with `DEFAULT_LLM_MODEL`.
- `FORCE_DEFAULT_LLM_MODEL=false`: use request `model` when provided, fallback to `DEFAULT_LLM_MODEL`.

## Real Request/Response Sample

Below is a real response captured from local call (FastAPI TestClient), not hand-written output.

### `/api/app/v1/words/extract`

Input:

```json
{
  "text": "The abduction case shocked the paradigm of modern society while ubiquitous systems remained stable.",
  "annotationMode": "SUPERSCRIPTS",
  "filterMode": "FREQUENCY",
  "frequency": "3000",
  "outputMode": "ANNOTATION",
  "lang": "en",
  "targetLanguage": "zh-CN"
}
```

Output:

```json
{
  "success": true,
  "requestId": "req_b7ccb57bb8314b0e",
  "data": {
    "translations": {
      "ubiquitous": "普遍存在的",
      "abduction": "绑架",
      "paradigm": "范式"
    },
    "items": [
      {
        "original": "ubiquitous",
        "translation": "无处不在的",
        "zipf": 3.42,
        "rank": "uncommon"
      },
      {
        "original": "abduction",
        "translation": "绑架",
        "zipf": 3.43,
        "rank": "uncommon"
      },
      {
        "original": "paradigm",
        "translation": "范例",
        "zipf": 3.61,
        "rank": "uncommon"
      }
    ],
    "engine": "wordfreq+dictionary-batch",
    "meta": {
      "engine": "wordfreq+dictionary-batch",
      "detectedSourceLanguage": "en",
      "requestedTargetLanguage": "zh-CN",
      "targetLanguage": "zh-CN",
      "filterMode": "FREQUENCY",
      "frequency": "3000",
      "thresholdZipf": 3.8,
      "sourceWordCount": 13,
      "candidateCount": 3,
      "maxWords": 120,
      "lang": "en",
      "latencyMs": 1189
    }
  }
}
```
