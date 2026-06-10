import asyncio
import io
import json
import os
import re
import time
import threading
import requests
import numpy as np
import sacrebleu
import jiwer

from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import List
from dotenv import load_dotenv
from fastapi import Form
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
from rapidocr import OCRVersion, RapidOCR
from deep_translator import GoogleTranslator
# from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
ocr = RapidOCR(
    params={
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
    }
)

DEFAULT_SOURCE_LANGUAGE = "ja"
DEFAULT_TARGET_LANGUAGE = "id"

TEST_MAX_IMAGE_SIDE = 2560
LIVE_MAX_IMAGE_SIDE = 2240

TRANSLATION_CACHE = {}
TRANSLATION_CACHE_LOCK = threading.Lock()
MAX_TRANSLATION_WORKERS = 4

LANG_CODE_MAP = {
    "google": {
        "en": "en", "ja": "ja", "zh": "zh-CN", "id": "id",
        "ko": "ko", "ru": "ru", "es": "es", "fr": "fr", "de": "de"
    },
    "azure": {
        "en": "en", "ja": "ja", "zh": "zh-Hans", "id": "id",
        "ko": "ko",  "ru": "ru", "es": "es", "fr": "fr", "de": "de"
    },
    "libre": {
        "en": "en", "ja": "ja", "zh": "zh", "id": "id",
        "ko": "ko",  "ru": "ru", "es": "es", "fr": "fr", "de": "de"
    },
}

SOURCE_SCRIPT_RE = {
    "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]"),
    "zh": re.compile(r"[\u4e00-\u9fff]"),
    "ko": re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]"),
    "ru": re.compile(r"[\u0400-\u04ff]"),
    "latin": re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")
}

NOISE_TOKENS = {
    "staff", "cast", "music", "skip", "auto", "log", "menu", "back", 
    "close", "ok", "cancel", "start", "loading", "hp", "mp", "ap",
    "atk", "def", "lv", "master", "servant", "party", "shop", "summon", 
    "event", "story", "quest", "clear", "next", "info", "detail"
}

load_dotenv()

LIVE_TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "google").lower()

LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "http://localhost:5000")
LIBRETRANSLATE_API_KEY = os.getenv("LIBRETRANSLATE_API_KEY", "")

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY", "")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")

class TranslateRequest(BaseModel):
    texts: List[str]
    source_lang: str = DEFAULT_SOURCE_LANGUAGE
    target_lang: str = DEFAULT_TARGET_LANGUAGE
    backend: str = "google"

def normalize_lang(lang: str) -> str:
    value = (lang or "").strip().lower()
    return value if value in LANG_CODE_MAP["google"] else DEFAULT_SOURCE_LANGUAGE

def normalize_ocr_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^\w\s\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u0400-\u04ff\u00C0-\u024F！？。、,.!?\'\"():・\-—「」『』（）【】：]",
        "",
        text,
    )
    return text.strip()

def should_keep_text(text: str, score: float, source_lang: str) -> bool:
    text = normalize_ocr_text(text)

    if score < 0.75 or len(text) <= 1:
        return False
    if text.isnumeric() or text.lower() in NOISE_TOKENS:
        return False
    
    # Ignore symbols and only accept alphanumeric 
    if re.fullmatch(r"[A-Z0-9]{2,}", text) or re.match(r'^[\d\s\.,!?@#\$%\^&\*\(\)\[\]\{\}\\\|\:\;\<\>\/\-\_]+$', text):
        return False

    if source_lang == "ja" and not SOURCE_SCRIPT_RE["ja"].search(text):
        return False
    if source_lang in {"zh", "zh-CN", "zh-Hans", "zh-TW"} and not SOURCE_SCRIPT_RE["zh"].search(text):
        return False
    if source_lang == "ko" and not SOURCE_SCRIPT_RE["ko"].search(text):
        return False
    if source_lang == "ru" and not SOURCE_SCRIPT_RE["ru"].search(text):
        return False
    if source_lang in {"en", "id", "es", "fr", "de"} and not SOURCE_SCRIPT_RE["latin"].search(text):
        return False

    return True

def preprocess_image(image_bytes: bytes, max_image_side: int) -> tuple[np.ndarray, float]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    scale = min(1.0, max_image_side / max(w, h))

    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return np.array(image), scale

def extract_raw_ocr_items(image_bytes: bytes, source_lang: str, max_image_side: int = TEST_MAX_IMAGE_SIDE):
    img_array, scale = preprocess_image(image_bytes, max_image_side=max_image_side)
    res = ocr(img_array)

    items = []
    if res.boxes is None or res.txts is None or res.scores is None:
        return items

    for box, text, score in zip(res.boxes, res.txts, res.scores):
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0

        text_str = str(text)

        if not should_keep_text(text_str, score_f, source_lang):
            continue

        items.append({
            "box": [[float(c) / scale for c in pt] for pt in box],
            "text": str(text),
            "ocrScore": round(score_f, 4),
        })

    return items

def box_height(box) -> float:
    ys = [pt[1] for pt in box]
    return max(ys) - min(ys)

def center_y(box) -> float:
    return sum(pt[1] for pt in box) / len(box)

def calculate_bounding_box(group):
    xs = [pt[0] for item in group for pt in item["box"]]
    ys = [pt[1] for item in group for pt in item["box"]]
    return [
        [min(xs), min(ys)],
        [max(xs), min(ys)],
        [max(xs), max(ys)],
        [min(xs), max(ys)],
    ]

def merge_into_paragraphs(items, source_lang: str, y_threshold_factor=0.7, x_gap_factor=2.5):
    if not items:
        return []
    
    list_pattern = re.compile(r"^(\d+[\)\.]|ex\.|・|-|\*)")

    for item in items:
        box = item["box"]
        item["top"] = min(pt[1] for pt in box)
        item["bottom"] = max(pt[1] for pt in box)
        item["left"] = min(pt[0] for pt in box)
        item["right"] = max(pt[0] for pt in box)
        item["height"] = item["bottom"] - item["top"]
        item["center_y"] = (item["top"] + item["bottom"]) / 2

    sorted_items = sorted(items, key=lambda x: (x["top"], x["left"]))

    blocks = []
    for item in sorted_items:
        placed = False
        is_list_item = bool(list_pattern.match(item["text"].strip()))

        for block in blocks:
            last_item = block[-1]
            
            # Similar font heights means in the same paragraph
            avg_height = sum(x["height"] for x in block) / len(block)
            if not (0.5 <= item["height"] / max(1, avg_height) <= 2.0):
                continue

            # Horizontal Proximity
            is_same_line = abs(item["center_y"] - last_item["center_y"]) < (avg_height * 0.5)
            x_gap = item["left"] - last_item["right"]
            is_close_horizontally = is_same_line and (0 <= x_gap <= avg_height * 2.5)

            # Vertical Alignment
            y_gap = item["top"] - last_item["bottom"]
            is_close_vertically = -avg_height <= y_gap <= (avg_height * 1.0)
            
            block_left = min(x["left"] for x in block)
            block_right = max(x["right"] for x in block)
            overlap_x = max(0, min(block_right, item["right"]) - max(block_left, item["left"]))
            min_width = min((block_right - block_left), (item["right"] - item["left"]))
            
            is_aligned = (overlap_x > min_width * 0.3) or (abs(item["left"] - block_left) < avg_height)

            if is_list_item and not is_same_line:
                is_vertically_stacked = False
            else:
                is_vertically_stacked = is_close_vertically and is_aligned

            if is_close_horizontally or is_vertically_stacked:
                block.append(item)
                placed = True
                break
        
        if not placed:
            blocks.append([item])
    
    merged = []
    join_char = "" if source_lang in {"ja", "zh", "zh-CN", "zh-Hans", "zh-TW"} else " "

    for block in blocks:
        block_avg_height = sum(x["height"] for x in block) / max(1, len(block))
        y_bucket_step = max(1, block_avg_height * 0.5)

        block.sort(key=lambda x: (x["center_y"] // y_bucket_step, x["left"]))
        
        merged_text = join_char.join(x["text"] for x in block).strip()
        merged_score = min(x["ocrScore"] for x in block)

        for k in ["top", "bottom", "left", "right", "height", "center_y"]:
            for x in block:
                x.pop(k, None)

        merged.append({
            "box": calculate_bounding_box(block),
            "text": merged_text,
            "ocrScore": merged_score,
        })
    
    return merged

def load_debug_font(size: int):
    candidates = [
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\arial.ttf"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def get_lang_code(backend: str, lang: str, default: str = "auto") -> str:
    return LANG_CODE_MAP.get(backend, {}).get(lang, default)

def wrap_text(text: str, width: int = 28) -> str:
    text = text or ""
    if len(text) <= width:
        return text
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))

def cache_get(backend: str, source_lang: str, target_lang: str, text: str):
    key = (backend, source_lang, target_lang, text)
    with TRANSLATION_CACHE_LOCK:
        return TRANSLATION_CACHE.get(key)

def cache_set(backend: str, source_lang: str, target_lang: str, text: str, translated: str):
    key = (backend, source_lang, target_lang, text)
    with TRANSLATION_CACHE_LOCK:
        TRANSLATION_CACHE[key] = translated

def translate_google(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    if not texts:
        return []

    src_code = get_lang_code("google", source_lang)
    tgt_code = get_lang_code("google", target_lang)

    def _translate_single(text: str) -> str:
        cleaned = normalize_ocr_text(text)
        if not cleaned:
            return ""

        cached = cache_get("google", source_lang, target_lang, cleaned)
        if cached is not None:
            return cached

        try:
            local_translator = GoogleTranslator(source=src_code, target=tgt_code)
            translated = (local_translator.translate(cleaned) or "").strip()
        except Exception:
            translated = cleaned

        cache_set("google", source_lang, target_lang, cleaned, translated)
        return translated

    with ThreadPoolExecutor(max_workers=MAX_TRANSLATION_WORKERS) as pool:
        return list(pool.map(_translate_single, texts))

def translate_azure(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    if not texts:
        return []

    if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_REGION:
        return texts

    azure_codes = {"en": "en", "ja": "ja", "zh": "zh-Hans", "id": "id"}
    frm = azure_codes.get(source_lang, source_lang)
    to = azure_codes.get(target_lang, target_lang)

    cleaned_texts = [normalize_ocr_text(t) for t in texts]
    results = [""] * len(texts)

    pending_indices = []
    pending_texts = []

    for idx, cleaned in enumerate(cleaned_texts):
        if not cleaned:
            results[idx] = ""
            continue

        cached = cache_get("azure", source_lang, target_lang, cleaned)
        if cached is not None:
            results[idx] = cached
        else:
            pending_indices.append(idx)
            pending_texts.append(cleaned)

    if not pending_texts:
        return results

    url = f"{AZURE_TRANSLATOR_ENDPOINT.rstrip('/')}/translate"
    params = {"api-version": "3.0", "from": frm, "to": to}
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json",
    }
    body = [{"Text": t} for t in pending_texts]

    try:
        resp = requests.post(url, params=params, headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        for idx, cleaned, item in zip(pending_indices, pending_texts, data):
            translated = item["translations"][0]["text"] if item.get("translations") else cleaned
            translated = (translated or "").strip()
            results[idx] = translated
            cache_set("azure", source_lang, target_lang, cleaned, translated)

        return results
    except Exception:
        return texts
    
def translate_libre(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    if not texts:
        return []

    src_code = get_lang_code("libre", source_lang, default="auto")
    tgt_code = get_lang_code("libre", target_lang, default="id")

    def _translate_single(text: str) -> str:
        cleaned = normalize_ocr_text(text)
        if not cleaned:
            return ""

        cached = cache_get("libre", source_lang, target_lang, cleaned)
        if cached is not None:
            return cached

        payload = {"q": cleaned, "source": src_code, "target": tgt_code, "format": "text"}
        if LIBRETRANSLATE_API_KEY:
            payload["api_key"] = LIBRETRANSLATE_API_KEY

        try:
            resp = requests.post(
                f"{LIBRETRANSLATE_URL.rstrip('/')}/translate",
                data=payload,
                timeout=10,
            )
            resp.raise_for_status()
            translated = (resp.json().get("translatedText", cleaned) or "").strip()
        except Exception:
            translated = cleaned

        cache_set("libre", source_lang, target_lang, cleaned, translated)
        return translated

    with ThreadPoolExecutor(max_workers=MAX_TRANSLATION_WORKERS) as pool:
        return list(pool.map(_translate_single, texts))

def translate_many(texts: List[str], source_lang: str, target_lang: str, backend: str | None = None) -> List[str]:
    if not texts:
        return []
    
    if source_lang.strip().lower() == target_lang.strip().lower():
        return texts
    
    backend = (backend or LIVE_TRANSLATION_BACKEND).lower()
    if backend not in {"google", "azure", "libre"}:
        backend = "google"

    if backend == "azure":
        return translate_azure(texts, source_lang, target_lang)
    if backend == "libre":
        return translate_libre(texts, source_lang, target_lang)
    return translate_google(texts, source_lang, target_lang)

def process_image_bytes(
    image_bytes: bytes,
    source_lang: str,
    target_lang: str,
    include_metrics: bool = False,
    max_image_side: int = LIVE_MAX_IMAGE_SIDE,
    backend: str | None = None,
):
    ocr_start = time.time()
    img_array, scale = preprocess_image(image_bytes, max_image_side=max_image_side)
    res = ocr(img_array)
    ocr_time = time.time() - ocr_start

    if res.boxes is None or res.txts is None or res.scores is None:
        return {"data": [], "metrics": {}} if include_metrics else []

    raw_items = []
    for box, text, score in zip(res.boxes, res.txts, res.scores):
        text = normalize_ocr_text(str(text))
        try:
            score = float(score)
        except Exception:
            score = 0.0

        if not should_keep_text(text, score, source_lang):
            continue

        is_punctuation = bool(re.fullmatch(r"[！？。、,.!?]+", text))
        if box_height(box) < 16 and len(text) <= 3 and not is_punctuation:
            continue

        raw_items.append({
            "box": [[(float(c) / scale) for c in pt] for pt in box],
            "text": text,
            "ocrScore": round(score, 4),
        })

    if not raw_items:
        return {"data": [], "metrics": {}} if include_metrics else []
    
    print("\n" + "="*50)
    print("RAW OCR DETECTIONS:")
    for item in raw_items:
         print(f"  [{item['ocrScore']:.4f}] {item['text']}")
    
    grouped_items = merge_into_paragraphs(raw_items, source_lang)
    texts_to_translate = [item["text"] for item in grouped_items]

    print("\nMERGED PARAGRAPHS (SENT TO API):")
    for idx, text in enumerate(texts_to_translate):
         print(f"  Block {idx + 1}: {text}")
    print("="*50 + "\n")

    trans_start = time.time()
    translations = translate_many(texts_to_translate, source_lang, target_lang, backend)
    trans_time = time.time() - trans_start

    output = []
    for item, trans in zip(grouped_items, translations):
        final_text = trans.strip() if trans else item["text"]
        if final_text:
            output.append({
                "box": item["box"],
                "text": item["text"],
                "translated": final_text,
                "ocrScore": item["ocrScore"],
            })

    if include_metrics:
        return {
            "data": output,
            "metrics": {
                "ocr_latency_sec": round(ocr_time, 4),
                "translation_latency_sec": round(trans_time, 4),
                "total_text_blocks": len(raw_items),
                "translated_blocks": len(output),
            }
        }

    return output

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = {
        "sourceLang": DEFAULT_SOURCE_LANGUAGE,
        "targetLang": DEFAULT_TARGET_LANGUAGE,
    }

    try:
        while True:
            message = await ws.receive()

            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "language_pair":
                        state["sourceLang"] = normalize_lang(payload.get("sourceLang"))
                        state["targetLang"] = normalize_lang(payload.get("targetLang"))
                        await ws.send_json({"type": "language_ack", **state})
                except Exception:
                    pass
                continue

            if message.get("bytes") is not None:
                image_bytes = message["bytes"]
                print(f"Live Translation Backend: {LIVE_TRANSLATION_BACKEND}")
                
                loop = asyncio.get_running_loop()
                func = partial(
                    process_image_bytes,
                    image_bytes,
                    state["sourceLang"],
                    state["targetLang"],
                    False,
                    LIVE_MAX_IMAGE_SIDE,
                    LIVE_TRANSLATION_BACKEND,
                )

                try:
                    output = await loop.run_in_executor(None, func)
                    await ws.send_json(output)
                except Exception as e:
                    print(f"Dropped frame: {e}")
                    pass

    except (WebSocketDisconnect, RuntimeError):
        return
    
@app.post("/test-image")
async def test_image(file: UploadFile = File(...)):

    data = await file.read()
    output = process_image_bytes(
        data,
        DEFAULT_SOURCE_LANGUAGE,
        DEFAULT_TARGET_LANGUAGE,
        include_metrics=True,
        max_image_side=TEST_MAX_IMAGE_SIDE,
        backend=LIVE_TRANSLATION_BACKEND,
    )
    return output

@app.post("/test-image-ocr")
async def test_image_ocr(file: UploadFile = File(...)):
    data = await file.read()
    items = extract_raw_ocr_items(data, DEFAULT_SOURCE_LANGUAGE, max_image_side=TEST_MAX_IMAGE_SIDE)
    return {"items": items}

@app.post("/test-image-ocr-visualization")
async def test_image_ocr_visualization(file: UploadFile = File(...)):
    data = await file.read()
    items = extract_raw_ocr_items(data, DEFAULT_SOURCE_LANGUAGE, max_image_side=TEST_MAX_IMAGE_SIDE)

    image = Image.open(io.BytesIO(data)).convert("RGBA")

    font_box = load_debug_font(36)
    font_header = load_debug_font(22)
    font_text = load_debug_font(34)

    legend_w = 1200
    canvas_h = max(image.height, max(600, len(items) * 110))
    canvas = Image.new("RGBA", (image.width + legend_w, canvas_h), (0, 0, 0, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for idx, item in enumerate(items, 1):
        pts = [tuple(map(float, pt)) for pt in item["box"]]
        draw.polygon(pts, outline=(255, 0, 0, 255), width=3)

        x = min(p[0] for p in pts)
        y = min(p[1] for p in pts)

        draw.text(
            (x, max(0, y - 40)),
            f"#{idx}",
            font=font_box,
            fill=(255, 255, 0, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    legend_x = image.width + 20
    y = 20

    for idx, item in enumerate(items, 1):
        header = f"#{idx}  score={item['ocrScore']:.4f}"
        text = wrap_text(item["text"], width=28)

        draw.multiline_text(
            (legend_x, y),
            header,
            font=font_header,
            fill=(0, 255, 255, 255),
            spacing=4,
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
        y += 26

        draw.multiline_text(
            (legend_x, y),
            text,
            font=font_text,
            fill=(255, 255, 255, 255),
            spacing=6,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        y += 80

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return StreamingResponse(out, media_type="image/png")

@app.post("/translate-test")
async def translate_test(req: TranslateRequest):
    translations = translate_many(req.texts, req.source_lang, req.target_lang, req.backend)
    return {
        "backend": req.backend,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "translations": translations,
    }

@app.post("/test-image-translate")
async def test_image_translate(
    file: UploadFile = File(...),
    source_lang: str = DEFAULT_SOURCE_LANGUAGE,
    target_lang: str = DEFAULT_TARGET_LANGUAGE,
    backend: str = "google",
):
    data = await file.read()

    items = extract_raw_ocr_items(data, source_lang, max_image_side=TEST_MAX_IMAGE_SIDE)
    texts = [item["text"] for item in items]
    translations = translate_many(texts, source_lang, target_lang, backend)

    output = []
    for item, translated in zip(items, translations):
        output.append({
            "box": item["box"],
            "text": item["text"],
            "translated": translated,
            "ocrScore": item["ocrScore"],
        })

    return {
        "backend": backend,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "items": output,
    }

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    ground_truth_ocr: str = Form(...),
    ground_truth_translation: str = Form(...),
    source_lang: str = Form(DEFAULT_SOURCE_LANGUAGE),
    target_lang: str = Form(DEFAULT_TARGET_LANGUAGE),
    backend: str = Form("google")
):
    data = await file.read()

    result = process_image_bytes(
        image_bytes=data,
        source_lang=source_lang,
        target_lang=target_lang,
        include_metrics=True,
        max_image_side=TEST_MAX_IMAGE_SIDE,
        backend=backend
    )

    pipeline_data = result.get("data", [])
    pipeline_metrics = result.get("metrics", {})

    predicted_ocr = " ".join([item["text"] for item in pipeline_data]).strip()
    predicted_translation = " ".join([item["translated"] for item in pipeline_data]).strip()

    # Character Error Rate (CER) - Lower is better
    try:
        cer_score = jiwer.cer(ground_truth_ocr, predicted_ocr)
    except Exception:
        cer_score = 1.0

    # BLEU Score - Higher is better
    try:
        bleu = sacrebleu.corpus_bleu([predicted_translation], [[ground_truth_translation]])
        bleu_score = bleu.score
    except Exception:
        bleu_score = 0.0

    return {
        "latency_metrics": {
            "ocr_processing_sec": pipeline_metrics.get("ocr_latency_sec", 0),
            "translation_api_sec": pipeline_metrics.get("translation_latency_sec", 0),
            "total_pipeline_sec": round(pipeline_metrics.get("ocr_latency_sec", 0) + pipeline_metrics.get("translation_latency_sec", 0), 4)
        },
        "accuracy_metrics": {
            "cer_score": round(cer_score, 4),
            "bleu_score": round(bleu_score, 2)
        },
        "debug_output": {
            "predicted_ocr": predicted_ocr,
            "predicted_translation": predicted_translation
        }
    }