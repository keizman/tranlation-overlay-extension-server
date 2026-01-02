"""
Translation Overlay Extension Server
A FastAPI proxy server with Redis caching for LLM translation requests.
"""

import hashlib
import json
import os
import gzip
import shutil
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SITE_AUTH_TOKEN = "YXBpLTEyMzQ1Ng=="  # Base64 encoded auth token
DEFAULT_LLM_ENDPOINT = "http://127.0.0.1:8317/v1/chat/completions"
DEFAULT_CACHE_TTL_DAYS = 7
CACHE_TTL_CONFIG_KEY = "tl_config:cache_ttl_days"  # Redis key for TTL config
LOG_DIR = Path("logs")
MAX_LOG_SIZE_MB = 300

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


def generate_cache_key(body: dict, user_level: str = "") -> str:
    """Generate cache key from request body messages array + user level."""
    # Hash messages array (core content) + user level
    messages = body.get("messages", [])
    content_str = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    
    # Include user level in the hash
    combined = f"{content_str}|level:{user_level}"
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


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
    site_auth = request.headers.get("site_auth", request.headers.get("site-auth", ""))
    site_api = request.headers.get("site_api", request.headers.get("site-api", ""))
    authorization = request.headers.get("Authorization", "")
    
    # 2. Validate site_auth
    if site_auth != SITE_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid site_auth")
    
    # 3. Determine target endpoint
    target_url = site_api if site_api else DEFAULT_LLM_ENDPOINT
    
    # 4. Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    # 5. Extract user level for cache key
    user_level = extract_user_level(body)
    
    # 6. Check cache
    cache_key = generate_cache_key(body, user_level)
    cached_response = get_cached_response(cache_key)
    
    if cached_response:
        print(f"[CACHE HIT] {cache_key} (level: {user_level})")
        return cached_response
    
    print(f"[CACHE MISS] {cache_key} (level: {user_level}) -> {target_url}")
    
    # 7. Prepare body for forwarding (remove x_ fields and inject /no-think)
    forward_body = {k: v for k, v in body.items() if not k.startswith("x_")}
    
    # Inject /no-think at the start of user message content (for DeepSeek etc.)
    if forward_body.get("messages"):
        for msg in forward_body["messages"]:
            if msg.get("role") == "user" and msg.get("content"):
                content = msg["content"]
                if not content.startswith("/no-think"):
                    msg["content"] = "/no-think\n" + content
    
    # 8. Forward request to LLM
    forward_headers = {
        "Content-Type": "application/json",
        "Authorization": authorization,
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
            response = await client.post(
                target_url,
                json=forward_body,
                headers=forward_headers
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timeout")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")
    
    # 9. Process response
    if response.status_code == 200:
        try:
            response_data = response.json()
            # Cache successful response
            set_cached_response(cache_key, response_data)
            # Log for vocabulary building
            log_request_response(body, response_data, cache_key)
            return response_data
        except Exception:
            return Response(content=response.text, status_code=response.status_code)
    else:
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
