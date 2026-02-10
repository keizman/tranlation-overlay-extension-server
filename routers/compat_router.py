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

GOOGLE_TRANSLATE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"

COMPAT_FREQUENCY_THRESHOLDS = {
    "500": 4.8,
    "1500": 4.2,
    "3000": 3.8,
    "5000": 3.5,
    "8000": 3.2,
}
ENGINE_NAME = "wordfreq+google-translate"


def extract_client_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return ""


def validate_outer_auth(request: Request) -> None:
    client_token = extract_client_token(request)
    if client_token != SITE_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")


def normalize_target_language(lang: str) -> str:
    cleaned = (lang or "").strip().lower()
    if not cleaned:
        return DEFAULT_COMPAT_TARGET_LANGUAGE

    mapping = {
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-hans": "zh-CN",
        "zh-tw": "zh-TW",
        "en-us": "en",
        "en-gb": "en",
    }
    return mapping.get(cleaned, lang.strip())


async def google_translate_text(
    client: httpx.AsyncClient, text: str, target_language: str
) -> Tuple[str, str]:
    params = {
        "client": "gtx",
        "sl": "auto",
        "tl": normalize_target_language(target_language),
        "dt": "t",
        "q": text,
    }
    try:
        response = await client.get(
            GOOGLE_TRANSLATE_ENDPOINT,
            params=params,
            timeout=20.0,
        )
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Translate API failed: {str(e)}")

    try:
        data = response.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Translate API returned invalid JSON")

    translated = ""
    if isinstance(data, list) and data and isinstance(data[0], list):
        for seg in data[0]:
            if isinstance(seg, list) and seg and isinstance(seg[0], str):
                translated += seg[0]

    if not translated.strip():
        translated = text

    detected = data[2] if isinstance(data, list) and len(data) > 2 else "unknown"
    if not isinstance(detected, str):
        detected = "unknown"

    return translated.strip(), detected


def tokenize_words_for_compat(text: str) -> List[str]:
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

    threshold = COMPAT_FREQUENCY_THRESHOLDS.get(
        req.frequency, COMPAT_FREQUENCY_THRESHOLDS["3000"]
    )
    normalized_target_language = normalize_target_language(req.targetLanguage)
    normalized_filter_mode = req.filterMode.upper()
    tokens = tokenize_words_for_compat(req.text)

    candidate_pairs: List[Tuple[str, float]] = []
    if normalized_filter_mode == "LEXICON":
        # Lexicon mode: keep a broader set for learning
        for word in tokens[: req.maxWords]:
            candidate_pairs.append((word, zipf_frequency(word, req.lang)))
    else:
        difficult_pairs = []
        for w in tokens:
            zf = zipf_frequency(w, req.lang)
            if zf < threshold:
                difficult_pairs.append((w, zf))
        difficult_pairs.sort(key=lambda x: x[1])
        candidate_pairs = difficult_pairs[: req.maxWords]

    candidate_words = [w for w, _ in candidate_pairs]
    items = []
    for word, zf in candidate_pairs:
        items.append(
            {
                "original": word,
                "translation": word,
                "zipf": round(zf, 2),
                "rank": zipf_to_rank_label(zf),
            }
        )

    latency_ms = int((time.perf_counter() - start_ts) * 1000)
    default_meta = {
        "engine": ENGINE_NAME,
        "detectedSourceLanguage": req.lang,
        "requestedTargetLanguage": req.targetLanguage,
        "targetLanguage": normalized_target_language,
        "filterMode": normalized_filter_mode,
        "frequency": req.frequency,
        "thresholdZipf": threshold,
        "sourceWordCount": len(tokens),
        "candidateCount": len(candidate_words),
        "maxWords": req.maxWords,
        "lang": req.lang,
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

    translations: Dict[str, str] = {}
    detected_source_language = req.lang
    async with httpx.AsyncClient() as client:
        for index, word in enumerate(candidate_words):
            translated, detected = await google_translate_text(
                client, word, req.targetLanguage
            )
            translations[word] = translated if translated else word
            items[index]["translation"] = translations[word]
            if detected and detected != "unknown":
                detected_source_language = detected

    latency_ms = int((time.perf_counter() - start_ts) * 1000)
    default_meta["detectedSourceLanguage"] = detected_source_language
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
