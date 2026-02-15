"""
Enladder-compatible API router (non-LLM implementation).
Provides:
- /api/app/v1/words/extract
"""

import os
import re
import time
import uuid
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from wordfreq import zipf_frequency

router = APIRouter(tags=["compat"])

SITE_AUTH_TOKEN = os.getenv("SITE_AUTH_TOKEN", "YXBpLTEyMzQ1Ng==")
DEFAULT_COMPAT_TARGET_LANGUAGE = os.getenv("DEFAULT_COMPAT_TARGET_LANGUAGE", "zh-CN")

DICTIONARY_SERVER_BASE_URL = os.getenv(
    "DICTIONARY_SERVER_BASE_URL", "http://127.0.0.1:9000"
).rstrip("/")
DICTIONARY_SERVER_BATCH_PATH = os.getenv(
    "DICTIONARY_SERVER_BATCH_PATH", "/api/v2/translate/batch"
)
DICTIONARY_SERVER_TIMEOUT_MS = int(os.getenv("DICTIONARY_SERVER_TIMEOUT_MS", "20000"))

COMPAT_FREQUENCY_THRESHOLDS = {
    "500": 4.8,
    "1500": 4.2,
    "3000": 3.8,
    "5000": 3.5,
    "8000": 3.2,
}

ERROR_CODE_UNSUPPORTED_LANGUAGE_PAIR = "UNSUPPORTED_LANGUAGE_PAIR"
ERROR_CODE_INVALID_LANGUAGE = "INVALID_LANGUAGE"
ENGINE_NAME = "wordfreq+dictionary-batch"


def extract_client_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return ""


def validate_outer_auth(request: Request) -> None:
    client_token = extract_client_token(request)
    if client_token != SITE_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")


def normalize_source_language(lang: str) -> str:
    cleaned = (lang or "").strip().lower()
    mapping = {
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "ja": "ja",
        "ko": "ko",
        "de": "de",
        "ru": "ru",
    }
    return mapping.get(cleaned, "")


def normalize_target_language(lang: str) -> str:
    cleaned = (lang or "").strip().lower()
    if not cleaned:
        cleaned = DEFAULT_COMPAT_TARGET_LANGUAGE.lower()

    mapping = {
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-hans": "zh-CN",
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "ja": "ja",
        "ko": "ko",
        "de": "de",
        "ru": "ru",
    }
    return mapping.get(cleaned, "")


def validate_language_pair(source_language: str, target_language: str) -> Optional[str]:
    if source_language == "en" and target_language == "zh-CN":
        return None
    if source_language == "en" and target_language in {"ja", "ko", "de", "ru"}:
        return None
    if source_language in {"ja", "ko", "de", "ru"} and target_language == "en":
        return None
    return f"Unsupported language pair: {source_language} -> {target_language}"


def tokenize_words_for_compat(text: str, language: str) -> List[str]:
    lang = (language or "").lower()
    if lang in {"en", "de"}:
        words = re.findall(r"[A-Za-zÄÖÜäöüß]+(?:'[A-Za-zÄÖÜäöüß]+)?", text)
        return sorted(set(word.lower() for word in words))
    if lang == "ru":
        words = re.findall(r"[А-Яа-яЁё]+", text)
        return sorted(set(words))
    if lang == "ko":
        words = re.findall(r"[가-힣]+", text)
        return sorted(set(words))
    if lang == "ja":
        words = re.findall(r"[ぁ-んァ-ンー一-龯]+", text)
        return sorted(set(words))
    # Fallback: keep previous English-like behavior for unknown languages.
    return sorted(set(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())))


def zipf_to_rank_label(zipf: float) -> str:
    if zipf >= 6.0:
        return "top100"
    if zipf >= 4.0:
        return "common"
    if zipf >= 3.0:
        return "uncommon"
    if zipf >= 2.0:
        return "rare"
    return "very_rare"


def safe_zipf_frequency(word: str, language: str) -> float:
    try:
        return zipf_frequency(word, language)
    except Exception:
        return 0.0


def build_batch_url() -> str:
    if DICTIONARY_SERVER_BATCH_PATH.startswith("/"):
        return f"{DICTIONARY_SERVER_BASE_URL}{DICTIONARY_SERVER_BATCH_PATH}"
    return f"{DICTIONARY_SERVER_BASE_URL}/{DICTIONARY_SERVER_BATCH_PATH}"


async def query_dictionary_batch(
    client: httpx.AsyncClient,
    words: List[str],
    source_language: str,
    target_language: str,
) -> Dict[str, str]:
    payload = {
        "targetLanguage": target_language,
        "items": [{"word": word, "language": source_language} for word in words],
    }

    url = build_batch_url()
    try:
        response = await client.post(
            url,
            json=payload,
            timeout=DICTIONARY_SERVER_TIMEOUT_MS / 1000.0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Dictionary batch request failed: {type(e).__name__}: {repr(e)}",
        )

    try:
        body = response.json()
    except Exception:
        raise HTTPException(
            status_code=502, detail="Dictionary batch returned invalid JSON"
        )

    if response.status_code == 422:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        code = error.get("code", ERROR_CODE_UNSUPPORTED_LANGUAGE_PAIR)
        message = error.get("message", "Unsupported language pair")
        if code == ERROR_CODE_UNSUPPORTED_LANGUAGE_PAIR:
            raise HTTPException(
                status_code=422,
                detail={"code": ERROR_CODE_UNSUPPORTED_LANGUAGE_PAIR, "message": message},
            )
        raise HTTPException(status_code=422, detail={"code": code, "message": message})

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Dictionary batch API failed: HTTP {response.status_code}",
        )

    if not isinstance(body, dict) or not body.get("success"):
        error = body.get("error", {}) if isinstance(body, dict) else {}
        code = error.get("code", "DICTIONARY_BATCH_ERROR")
        message = error.get("message", "Dictionary batch API returned failure")
        status = 422 if code == ERROR_CODE_UNSUPPORTED_LANGUAGE_PAIR else 502
        raise HTTPException(status_code=status, detail={"code": code, "message": message})

    results = body.get("data", {}).get("results", [])
    translation_map: Dict[str, str] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        translation = item.get("translation", "")
        translation_map[word] = translation if isinstance(translation, str) else ""

    return translation_map


class CompatWordsExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    annotationMode: str = Field("SUPERSCRIPTS")
    filterMode: str = Field("FREQUENCY")
    frequency: str = Field("3000")
    outputMode: str = Field("ANNOTATION")
    lang: str = Field("en", min_length=2, max_length=10)
    targetLanguage: str = Field(
        DEFAULT_COMPAT_TARGET_LANGUAGE, min_length=2, max_length=20
    )
    maxWords: int = Field(120, ge=1, le=300)
    model: Optional[str] = Field(default=None)


@router.post("/api/app/v1/words/extract")
async def compat_words_extract(request: Request, req: CompatWordsExtractRequest):
    validate_outer_auth(request)
    start_ts = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:16]}"

    source_language = normalize_source_language(req.lang)
    target_language = normalize_target_language(req.targetLanguage)

    if not source_language or not target_language:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ERROR_CODE_INVALID_LANGUAGE,
                "message": "Invalid language. Supported source: en/ja/ko/de/ru. Supported target: zh-CN/en/ja/ko/de/ru.",
            },
        )

    pair_error = validate_language_pair(source_language, target_language)
    if pair_error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": ERROR_CODE_UNSUPPORTED_LANGUAGE_PAIR,
                "message": pair_error,
            },
        )

    threshold = COMPAT_FREQUENCY_THRESHOLDS.get(
        req.frequency, COMPAT_FREQUENCY_THRESHOLDS["3000"]
    )
    normalized_filter_mode = req.filterMode.upper()
    tokens = tokenize_words_for_compat(req.text, source_language)

    candidate_pairs: List[Tuple[str, float]] = []
    if normalized_filter_mode == "LEXICON":
        for word in tokens[: req.maxWords]:
            candidate_pairs.append((word, safe_zipf_frequency(word, source_language)))
    else:
        difficult_pairs = []
        for word in tokens:
            zf = safe_zipf_frequency(word, source_language)
            if zf < threshold:
                difficult_pairs.append((word, zf))
        difficult_pairs.sort(key=lambda x: x[1])
        candidate_pairs = difficult_pairs[: req.maxWords]

    candidate_words = [word for word, _ in candidate_pairs]
    items = []
    for word, zf in candidate_pairs:
        items.append(
            {
                "original": word,
                "translation": "",
                "zipf": round(zf, 2),
                "rank": zipf_to_rank_label(zf),
            }
        )

    latency_ms = int((time.perf_counter() - start_ts) * 1000)
    default_meta = {
        "engine": ENGINE_NAME,
        "detectedSourceLanguage": source_language,
        "requestedTargetLanguage": req.targetLanguage,
        "targetLanguage": target_language,
        "filterMode": normalized_filter_mode,
        "frequency": req.frequency,
        "thresholdZipf": threshold,
        "sourceWordCount": len(tokens),
        "candidateCount": len(candidate_words),
        "maxWords": req.maxWords,
        "lang": source_language,
        "latencyMs": latency_ms,
    }

    if not candidate_words:
        return {
            "success": True,
            "requestId": request_id,
            "data": {
                "translations": {},
                "items": [],
                "engine": ENGINE_NAME,
                "meta": default_meta,
            },
        }

    async with httpx.AsyncClient(trust_env=False) as client:
        translation_map = await query_dictionary_batch(
            client=client,
            words=candidate_words,
            source_language=source_language,
            target_language=target_language,
        )

    translations: Dict[str, str] = {}
    for index, word in enumerate(candidate_words):
        translated = translation_map.get(word, "")
        translations[word] = translated
        items[index]["translation"] = translated

    latency_ms = int((time.perf_counter() - start_ts) * 1000)
    default_meta["latencyMs"] = latency_ms

    return {
        "success": True,
        "requestId": request_id,
        "data": {
            "translations": translations,
            "items": items,
            "engine": ENGINE_NAME,
            "meta": default_meta,
        },
    }
