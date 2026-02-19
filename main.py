"""
Translation Overlay Extension Server
A FastAPI proxy server with Redis caching for LLM translation requests.
"""

import hashlib
import json
import os
import gzip
import shutil
from threading import Lock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import redis
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Translation Overlay Server", version="1.0.0")

# Register routers (modular API endpoints)
from routers.compat_router import router as compat_router
app.include_router(compat_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - read from environment variables with defaults
SITE_AUTH_TOKEN = os.getenv("SITE_AUTH_TOKEN", "YXBpLTEyMzQ1Ng==")  # Client -> Server auth
LLM_SITE_AUTH = os.getenv("LLM_SITE_AUTH", "")  # Server -> LLM auth (if set, overrides header)
DEFAULT_LLM_ENDPOINT = os.getenv("DEFAULT_LLM_ENDPOINT", "http://127.0.0.1:8317/v1/chat/completions")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash-lite").strip()
FORCE_DEFAULT_LLM_MODEL = os.getenv("FORCE_DEFAULT_LLM_MODEL", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_FIRST_REQUEST_ENABLED = os.getenv("FAST_FIRST_REQUEST_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
FAST_LLM_ENDPOINT = os.getenv("FAST_LLM_ENDPOINT", "").strip()
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "").strip()
FAST_LLM_SITE_AUTHS = [
    item.strip() for item in os.getenv("FAST_LLM_SITE_AUTHS", "").split(",") if item.strip()
]
FAST_LLM_KEY_ROTATION = os.getenv("FAST_LLM_KEY_ROTATION", "round_robin").strip().lower()
FAST_LLM_FALLBACK_TO_DEFAULT_ON_ERROR = os.getenv(
    "FAST_LLM_FALLBACK_TO_DEFAULT_ON_ERROR", "true"
).strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_CACHE_TTL_DAYS = int(os.getenv("DEFAULT_CACHE_TTL_DAYS", "30"))
CACHE_TTL_CONFIG_KEY = "tl_config:cache_ttl_days"  # Redis key for TTL config
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
MAX_LOG_SIZE_MB = int(os.getenv("MAX_LOG_SIZE_MB", "300"))
FAST_LLM_AUTH_INDEX = 0
FAST_LLM_AUTH_LOCK = Lock()

# Redis connection
REDIS_CONN_STRING = os.getenv("REDIS_CONN_STRING", "redis://localhost:6379/0")
redis_client: Optional[redis.Redis] = None

try:
    redis_client = redis.from_url(REDIS_CONN_STRING, decode_responses=True)
    redis_client.ping()
    print(f"[INFO] Redis connected: {REDIS_CONN_STRING.split('@')[-1] if '@' in REDIS_CONN_STRING else REDIS_CONN_STRING}")
    
    # Initialize TTL config on startup: read from Redis, write default if not exists
    existing_ttl = redis_client.get(CACHE_TTL_CONFIG_KEY)
    if existing_ttl is not None:
        print(f"[INFO] Cache TTL config loaded from Redis: {existing_ttl} days")
    else:
        redis_client.set(CACHE_TTL_CONFIG_KEY, str(DEFAULT_CACHE_TTL_DAYS))
        print(f"[INFO] Cache TTL config initialized to default: {DEFAULT_CACHE_TTL_DAYS} days")
except Exception as e:
    print(f"[WARN] Redis connection failed: {e}. Running without cache.")
    redis_client = None

print(
    f"[INFO] LLM model policy: DEFAULT_LLM_MODEL={DEFAULT_LLM_MODEL} "
    f"FORCE_DEFAULT_LLM_MODEL={FORCE_DEFAULT_LLM_MODEL}"
)
print(
    f"[INFO] FAST model policy: enabled={FAST_FIRST_REQUEST_ENABLED} "
    f"fast_model={FAST_LLM_MODEL or '<inherit>'} "
    f"fast_endpoint={'set' if FAST_LLM_ENDPOINT else 'inherit'} "
    f"fast_keys={len(FAST_LLM_SITE_AUTHS)} "
    f"fallback_to_default={FAST_LLM_FALLBACK_TO_DEFAULT_ON_ERROR}"
)


def get_cache_ttl_days() -> int:
    """Get cache TTL days from Redis config, fallback to default."""
    if not redis_client:
        return DEFAULT_CACHE_TTL_DAYS
    try:
        ttl_str = redis_client.get(CACHE_TTL_CONFIG_KEY)
        if ttl_str is not None:
            ttl_days = int(ttl_str)
            return ttl_days if ttl_days >= 0 else 0  # 0 means never expire
        return DEFAULT_CACHE_TTL_DAYS
    except Exception as e:
        print(f"[WARN] Failed to get TTL config: {e}")
        return DEFAULT_CACHE_TTL_DAYS


def set_cache_ttl_days(days: int) -> bool:
    """Set cache TTL days in Redis and refresh all existing cache TTLs."""
    if not redis_client:
        return False
    try:
        redis_client.set(CACHE_TTL_CONFIG_KEY, str(days))
        
        # Refresh TTL for all existing cache entries
        refresh_all_cache_ttls(days)
        
        print(f"[INFO] Cache TTL updated to {days} days")
        return True
    except Exception as e:
        print(f"[WARN] Failed to set TTL config: {e}")
        return False


def refresh_all_cache_ttls(days: int):
    """Refresh TTL for all existing cache entries."""
    if not redis_client:
        return
    try:
        # Find all cache keys
        cursor = 0
        refreshed_count = 0
        ttl_seconds = days * 24 * 60 * 60 if days > 0 else -1  # -1 means persist
        
        while True:
            cursor, keys = redis_client.scan(cursor, match="tl_cache:*", count=100)
            for key in keys:
                if days == 0:
                    # 0 means never expire - persist the key
                    redis_client.persist(key)
                else:
                    redis_client.expire(key, ttl_seconds)
                refreshed_count += 1
            
            if cursor == 0:
                break
        
        print(f"[INFO] Refreshed TTL for {refreshed_count} cache entries")
    except Exception as e:
        print(f"[WARN] Failed to refresh cache TTLs: {e}")


import re
import unicodedata


def normalize_for_cache(text: str) -> str:
    """
    Normalize text content before cache hash calculation.
    Removes dynamic content that may change between page loads but doesn't affect translation.
    """
    if not text:
        return ""
    
    normalized = text
    
    # 1. Unicode normalization (NFC form)
    normalized = unicodedata.normalize('NFC', normalized)
    
    # 2. Remove zero-width characters
    normalized = re.sub(r'[\u200B\uFEFF\u00AD\u200C\u200D\u2060]', '', normalized)
    
    # 3. Remove dynamic timestamps (Chinese)
    normalized = re.sub(
        r'\d+\s*(分钟|小时|天|秒|周|月|年)前|刚刚|刚才',
        '',
        normalized
    )
    
    # 4. Remove dynamic timestamps (English)
    normalized = re.sub(
        r'\d+\s*(hours?|minutes?|days?|seconds?|weeks?|months?|years?)\s*ago|just now|a moment ago',
        '',
        normalized,
        flags=re.IGNORECASE
    )
    
    # 5. Remove stats/counts (Chinese)
    normalized = re.sub(
        r'\d+\.?\d*[kKmMwW万亿]?\s*(赞|喜欢|评论|回复|浏览|阅读|收藏|转发|分享|播放|观看)',
        '',
        normalized
    )
    
    # 6. Remove stats/counts (English)
    normalized = re.sub(
        r'\d+\.?\d*[kKmM]?\s*(likes?|views?|comments?|replies?|reads?|shares?|plays?)',
        '',
        normalized,
        flags=re.IGNORECASE
    )
    
    # 7. Remove progress indicators
    normalized = re.sub(
        r'第?\d+[章节篇回话集]?\s*[\/\\|]\s*(共|of)?\s*\d+|[(（]\d+[\/\\|]\d+[)）]|\d+(\.\d+)?%',
        '',
        normalized,
        flags=re.IGNORECASE
    )
    
    # 8. Normalize full-width punctuation to half-width (common ones)
    fullwidth_map = {
        '：': ':',
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
    }
    for fw, hw in fullwidth_map.items():
        normalized = normalized.replace(fw, hw)
    
    # 9. Normalize whitespace (collapse multiple spaces/newlines to single space)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def resolve_effective_model(body: dict) -> tuple[str, str]:
    """
    Resolve effective model for upstream LLM call.
    Returns (effective_model, model_source).
    model_source in: request | env_default | env_force
    """
    request_model = body.get("model")
    if isinstance(request_model, str) and request_model.strip():
        if FORCE_DEFAULT_LLM_MODEL:
            return DEFAULT_LLM_MODEL, "env_force"
        return request_model.strip(), "request"
    return DEFAULT_LLM_MODEL, "env_default"


def parse_page_request_seq(body: dict) -> Optional[int]:
    """Parse x_page_request_seq from extension metadata."""
    value = body.get("x_page_request_seq")
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            parsed = int(value)
            return parsed if parsed > 0 else None
    return None


def should_use_fast_lane(body: dict) -> bool:
    """Only first request per page can use FAST lane when enabled."""
    if not FAST_FIRST_REQUEST_ENABLED:
        return False
    return parse_page_request_seq(body) == 1


def next_fast_llm_auth() -> tuple[str, str]:
    """Pick next FAST LLM key using configured rotation policy."""
    if not FAST_LLM_SITE_AUTHS:
        return "", "missing"

    if FAST_LLM_KEY_ROTATION == "round_robin":
        global FAST_LLM_AUTH_INDEX
        with FAST_LLM_AUTH_LOCK:
            selected_index = FAST_LLM_AUTH_INDEX % len(FAST_LLM_SITE_AUTHS)
            FAST_LLM_AUTH_INDEX += 1
        return FAST_LLM_SITE_AUTHS[selected_index], f"fast_pool[{selected_index}]"

    return FAST_LLM_SITE_AUTHS[0], "fast_pool[0]"


def resolve_primary_route(body: dict, default_route: dict) -> dict:
    """Resolve whether to use default lane or FAST lane for current request."""
    route = default_route.copy()

    if not should_use_fast_lane(body):
        route["lane_reason"] = "not_first_request"
        return route

    has_fast_override = bool(FAST_LLM_ENDPOINT or FAST_LLM_MODEL or FAST_LLM_SITE_AUTHS)
    if not has_fast_override:
        route["lane_reason"] = "fast_unconfigured"
        return route

    route["lane"] = "fast"
    route["lane_reason"] = "first_request"

    if FAST_LLM_ENDPOINT:
        route["target_url"] = FAST_LLM_ENDPOINT

    if FAST_LLM_MODEL:
        route["model"] = FAST_LLM_MODEL
        route["model_source"] = "env_fast"

    fast_auth, fast_auth_source = next_fast_llm_auth()
    if fast_auth:
        route["llm_auth"] = fast_auth
        route["auth_source"] = fast_auth_source

    return route


def build_forward_body(body: dict, model: str) -> dict:
    """Build upstream payload by removing extension metadata and normalizing fields."""
    forward_body = {k: v for k, v in body.items() if not k.startswith("x_")}
    forward_body["model"] = model
    forward_body["stream"] = False

    if forward_body.get("messages"):
        normalized_messages = []
        for msg in forward_body["messages"]:
            if not isinstance(msg, dict):
                normalized_messages.append(msg)
                continue

            normalized_msg = msg.copy()
            if normalized_msg.get("role") == "user" and normalized_msg.get("content"):
                content = normalized_msg["content"]
                if not content.startswith("/no-think"):
                    normalized_msg["content"] = "/no-think\n" + content
            normalized_messages.append(normalized_msg)
        forward_body["messages"] = normalized_messages

    return forward_body


async def forward_to_llm(route: dict, forward_body: dict) -> httpx.Response:
    """Forward request to LLM endpoint selected by route policy."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {route['llm_auth']}",
    }

    print(
        f"[LLM] lane={route['lane']} target={route['target_url']} model={forward_body.get('model')} "
        f"model_source={route['model_source']} auth_source={route['auth_source']}"
    )

    try:
        async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
            return await client.post(
                route["target_url"],
                json=forward_body,
                headers=headers,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timeout")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")


def log_llm_error(response: httpx.Response, route: dict):
    print(f"[LLM ERROR] Status: {response.status_code}")
    print(f"[LLM ERROR] Lane: {route['lane']}")
    print(f"[LLM ERROR] Target: {route['target_url']}")
    print(f"[LLM ERROR] Auth source: {route['auth_source']}")
    print(f"[LLM ERROR] Response: {response.text[:200]}")


def generate_cache_key(
    body: dict,
    user_level: str = "",
    model: str = "",
    lane: str = "default",
) -> str:
    """Generate cache key from request body messages + user level + model + lane."""
    # Hash messages array (core content) + user level
    messages = body.get("messages", [])
    
    # Normalize message content before hashing
    normalized_messages = []
    for msg in messages:
        normalized_msg = msg.copy()
        if "content" in normalized_msg and isinstance(normalized_msg["content"], str):
            normalized_msg["content"] = normalize_for_cache(normalized_msg["content"])
        normalized_messages.append(normalized_msg)
    
    content_str = json.dumps(normalized_messages, sort_keys=True, ensure_ascii=False)
    
    # Include user level + model + lane in the hash
    combined = f"{content_str}|level:{user_level}|model:{model}|lane:{lane}"
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def is_valid_response_content(response_data: dict) -> bool:
    """
    Validate that response content is not empty or invalid.
    Checks for: null, empty string, "Empty string.", zero-width spaces, etc.
    """
    try:
        choices = response_data.get("choices", [])
        if not choices:
            return False
        
        message = choices[0].get("message", {})
        content = message.get("content")
        
        # Check for null
        if content is None:
            return False
        
        # Check for empty or whitespace-only string
        if not isinstance(content, str):
            return False
        
        # Strip regular whitespace
        stripped = content.strip()
        if not stripped:
            return False
        
        # Check for known invalid responses
        invalid_patterns = [
            "Empty string.",
            "​​",  # Zero-width spaces
            "\u200b",  # Zero-width space
            "\u200c",  # Zero-width non-joiner
            "\u200d",  # Zero-width joiner
            "\ufeff",  # BOM
        ]
        
        for pattern in invalid_patterns:
            if stripped == pattern or stripped.replace(pattern, "") == "":
                return False
        
        return True
    except Exception:
        return False


def extract_user_level(body: dict) -> str:
    """Extract user level from request body or headers."""
    # Check for x_user_level in body (will be added by extension)
    user_level = body.get("x_user_level", "")
    return str(user_level) if user_level else "default"


def get_cached_response(cache_key: str) -> Optional[dict]:
    """Get cached response from Redis."""
    if not redis_client:
        return None
    try:
        cached = redis_client.get(f"tl_cache:{cache_key}")
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"[WARN] Cache read error: {e}")
    return None


def set_cached_response(cache_key: str, response_data: dict):
    """Store response in Redis cache."""
    if not redis_client:
        return
    try:
        ttl_days = get_cache_ttl_days()
        
        if ttl_days == 0:
            # 0 means never expire
            redis_client.set(
                f"tl_cache:{cache_key}",
                json.dumps(response_data, ensure_ascii=False)
            )
        else:
            ttl_seconds = ttl_days * 24 * 60 * 60
            redis_client.setex(
                f"tl_cache:{cache_key}",
                ttl_seconds,
                json.dumps(response_data, ensure_ascii=False)
            )
    except Exception as e:
        print(f"[WARN] Cache write error: {e}")


def log_request_response(request_body: dict, response_body: dict, cache_key: str):
    """Log request/response to daily JSON file with auto-compression."""
    LOG_DIR.mkdir(exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"{today}.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "cache_key": cache_key,
        "request": request_body,
        "response": response_body
    }
    
    # Append to daily log file
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Log write error: {e}")
    
    # Check total log size and compress if needed
    check_and_compress_logs()


def check_and_compress_logs():
    """Compress logs if total size exceeds MAX_LOG_SIZE_MB."""
    try:
        total_size = sum(f.stat().st_size for f in LOG_DIR.glob("*.json"))
        if total_size > MAX_LOG_SIZE_MB * 1024 * 1024:
            # Compress all JSON files except today's
            today = datetime.now().strftime("%Y-%m-%d")
            for json_file in LOG_DIR.glob("*.json"):
                if today not in json_file.name:
                    gz_file = json_file.with_suffix(".json.gz")
                    with open(json_file, "rb") as f_in:
                        with gzip.open(gz_file, "wb", compresslevel=9) as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    json_file.unlink()
                    print(f"[INFO] Compressed {json_file.name}")
    except Exception as e:
        print(f"[WARN] Log compression error: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with caching."""

    # 1. Extract headers
    # - Authorization: Bearer <server_api_key> - used to validate client request
    # - site_auth: <llm_api_key> - used to authenticate with LLM
    # - site_api: <llm_endpoint> - LLM endpoint URL
    authorization = request.headers.get("Authorization", "")
    site_auth = request.headers.get("site_auth", request.headers.get("site-auth", ""))
    site_api = request.headers.get("site_api", request.headers.get("site-api", ""))

    # 2. Validate client request using Authorization Bearer token
    client_token = ""
    if authorization.startswith("Bearer "):
        client_token = authorization[7:]  # Remove "Bearer " prefix

    if client_token != SITE_AUTH_TOKEN:
        print(f"[AUTH FAIL] Client token: '{client_token}'")
        print(f"[AUTH FAIL] Expected: '{SITE_AUTH_TOKEN}'")
        raise HTTPException(status_code=401, detail=f"Unauthorized: Invalid API key")

    print(f"[AUTH OK] Client authenticated, site_auth={'env' if LLM_SITE_AUTH else ('header' if site_auth else 'none')}, site_api={site_api or 'default'}")

    # 3. Determine default LLM endpoint and auth
    # Priority: env LLM_SITE_AUTH > header site_auth > client_token
    default_target_url = site_api if site_api else DEFAULT_LLM_ENDPOINT
    default_llm_auth = LLM_SITE_AUTH if LLM_SITE_AUTH else (site_auth if site_auth else client_token)

    # 4. Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 5. Resolve default route + optional FAST lane
    user_level = extract_user_level(body)
    effective_model, model_source = resolve_effective_model(body)
    default_route = {
        "lane": "default",
        "lane_reason": "default_lane",
        "target_url": default_target_url,
        "llm_auth": default_llm_auth,
        "model": effective_model,
        "model_source": model_source,
        "auth_source": "env" if LLM_SITE_AUTH else ("header" if site_auth else "client"),
    }
    active_route = resolve_primary_route(body, default_route)
    body["model"] = active_route["model"]

    # 6. Check cache
    cache_key = generate_cache_key(
        body,
        user_level,
        active_route["model"],
        active_route["lane"],
    )
    cached_response = get_cached_response(cache_key)

    if cached_response:
        # Validate cached content is not empty/invalid
        if is_valid_response_content(cached_response):
            print(
                f"⚡ [CACHE HIT] key={cache_key[:16]}... level={user_level} "
                f"lane={active_route['lane']} model={active_route['model']} "
                f"model_source={active_route['model_source']} | Returning cached response"
            )
            return cached_response
        else:
            print(f"⚠️ [CACHE INVALID] key={cache_key[:16]}... | Cached content is empty, fetching fresh")

    # Debug: print text preview for cache miss analysis
    messages = body.get("messages", [])
    user_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")[:80]
            break
    print(
        f"[CACHE MISS] key={cache_key} level={user_level} lane={active_route['lane']} "
        f"lane_reason={active_route['lane_reason']} model={active_route['model']} "
        f"model_source={active_route['model_source']}"
    )
    print(f"[CACHE MISS] text preview: \"{user_msg}...\"")
    print(f"[CACHE MISS] target: {active_route['target_url']}")

    # 7. Forward request (with optional fallback from FAST -> default)
    response: Optional[httpx.Response] = None
    used_fallback = False
    forward_body = build_forward_body(body, active_route["model"])

    try:
        response = await forward_to_llm(active_route, forward_body)
    except HTTPException as first_call_error:
        can_fallback = (
            active_route["lane"] == "fast" and FAST_LLM_FALLBACK_TO_DEFAULT_ON_ERROR
        )
        if not can_fallback:
            raise first_call_error

        used_fallback = True
        active_route = default_route.copy()
        body["model"] = active_route["model"]
        fallback_cache_key = generate_cache_key(
            body,
            user_level,
            active_route["model"],
            active_route["lane"],
        )
        fallback_cached = get_cached_response(fallback_cache_key)
        if fallback_cached and is_valid_response_content(fallback_cached):
            print(
                f"⚡ [CACHE HIT] key={fallback_cache_key[:16]}... level={user_level} "
                f"lane={active_route['lane']} model={active_route['model']} "
                f"model_source={active_route['model_source']} | Returning fallback cached response"
            )
            return fallback_cached

        print("[LLM FALLBACK] primary route exception, retrying default lane")
        forward_body = build_forward_body(body, active_route["model"])
        response = await forward_to_llm(active_route, forward_body)
        cache_key = fallback_cache_key

    if (
        response is not None
        and response.status_code != 200
        and active_route["lane"] == "fast"
        and FAST_LLM_FALLBACK_TO_DEFAULT_ON_ERROR
    ):
        log_llm_error(response, active_route)
        used_fallback = True

        active_route = default_route.copy()
        body["model"] = active_route["model"]
        fallback_cache_key = generate_cache_key(
            body,
            user_level,
            active_route["model"],
            active_route["lane"],
        )
        fallback_cached = get_cached_response(fallback_cache_key)
        if fallback_cached and is_valid_response_content(fallback_cached):
            print(
                f"⚡ [CACHE HIT] key={fallback_cache_key[:16]}... level={user_level} "
                f"lane={active_route['lane']} model={active_route['model']} "
                f"model_source={active_route['model_source']} | Returning fallback cached response"
            )
            return fallback_cached

        print("[LLM FALLBACK] primary route non-200, retrying default lane")
        forward_body = build_forward_body(body, active_route["model"])
        response = await forward_to_llm(active_route, forward_body)
        cache_key = fallback_cache_key

    if response is None:
        raise HTTPException(status_code=500, detail="LLM request failed unexpectedly")

    # 8. Process response
    if response.status_code == 200:
        try:
            response_data = response.json()

            # Validate content is not empty before caching
            if is_valid_response_content(response_data):
                # Cache successful response
                set_cached_response(cache_key, response_data)
                # Log for vocabulary building
                log_request_response(body, response_data, cache_key)
            else:
                print(f"[SKIP CACHE] Empty or invalid content detected, not caching")

            if used_fallback:
                print("[LLM FALLBACK] completed with default lane response")
            return response_data
        except Exception:
            return Response(content=response.text, status_code=response.status_code)
    else:
        # Log LLM error details
        log_llm_error(response, active_route)
        # Return error as-is
        return Response(
            content=response.text,
            status_code=response.status_code,
            media_type="application/json"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "healthy",
        "redis": redis_status,
        "cache_ttl_days": get_cache_ttl_days()
    }


@app.get("/config/cache-ttl")
async def get_cache_ttl():
    """Get current cache TTL configuration."""
    return {
        "cache_ttl_days": get_cache_ttl_days(),
        "description": "0 means never expire"
    }


@app.post("/config/cache-ttl")
async def update_cache_ttl(request: Request):
    """Update cache TTL and refresh all existing cache entries."""
    try:
        data = await request.json()
        days = int(data.get("days", DEFAULT_CACHE_TTL_DAYS))
        
        if days < 0:
            raise HTTPException(status_code=400, detail="days must be >= 0")
        
        success = set_cache_ttl_days(days)
        
        if success:
            return {
                "success": True,
                "cache_ttl_days": days,
                "message": f"TTL updated to {days} days. All existing cache entries refreshed."
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update TTL")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid days value")


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    if not redis_client:
        return {"error": "Redis not connected"}
    
    try:
        cursor = 0
        count = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match="tl_cache:*", count=100)
            count += len(keys)
            if cursor == 0:
                break
        
        return {
            "cache_entries": count,
            "cache_ttl_days": get_cache_ttl_days()
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
