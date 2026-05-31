import asyncio
import io
import json
import threading
from functools import lru_cache
from typing import Dict, List, Tuple
import cv2

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
ocr = RapidOCR()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_SOURCE_LANG = "ja"
DEFAULT_TARGET_LANG = "en"

SUPPORTED_LANGS = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "id": "Bahasa Indonesia",
}

DIRECT_MODEL_IDS: Dict[Tuple[str, str], str] = {
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("id", "en"): "Helsinki-NLP/opus-mt-id-en",
    ("en", "id"): "Helsinki-NLP/opus-mt-en-id",
}

TRANSLATION_CACHE: Dict[Tuple[str, str, str], str] = {}
TRANSLATION_CACHE_LOCK = threading.Lock()


def normalize_lang(lang: str) -> str:
    value = (lang or "").strip().lower()

    if value not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language code: {lang}")

    return value


def resolve_steps(source_lang: str, target_lang: str) -> List[Tuple[str, str]]:
    source_lang = normalize_lang(source_lang)
    target_lang = normalize_lang(target_lang)

    if source_lang == target_lang:
        return []

    if (source_lang, target_lang) in DIRECT_MODEL_IDS:
        return [(source_lang, target_lang)]

    # Translate to English first
    if source_lang != "en" and target_lang != "en":
        return [(source_lang, "en"), ("en", target_lang)]

    raise ValueError(f"No Marian route configured for {source_lang} -> {target_lang}")


@lru_cache(maxsize=8)
def load_model(model_id: str):
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def translate_step_batch(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    if not texts:
        return []

    model_id = DIRECT_MODEL_IDS[(source_lang, target_lang)]
    tokenizer, model = load_model(model_id)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            num_beams=1,
            max_new_tokens=128,
        )

    return [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]


def translate_pipeline(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    steps = resolve_steps(source_lang, target_lang)
    if not steps:
        return texts

    current_texts = texts
    for step_source, step_target in steps:
        current_texts = translate_step_batch(current_texts, step_source, step_target)
    return current_texts


def translate_texts_with_cache(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    source_lang = normalize_lang(source_lang)
    target_lang = normalize_lang(target_lang)

    if source_lang == target_lang:
        return texts

    final_results: List[str] = [""] * len(texts)
    pending_indices: List[int] = []
    pending_texts: List[str] = []

    for idx, raw_text in enumerate(texts):
        text = (raw_text or "").strip()
        if not text:
            final_results[idx] = ""
            continue

        cache_key = (source_lang, target_lang, text)
        with TRANSLATION_CACHE_LOCK:
            cached = TRANSLATION_CACHE.get(cache_key)

        if cached is not None:
            final_results[idx] = cached
        else:
            pending_indices.append(idx)
            pending_texts.append(text)

    if pending_texts:
        translated_pending = translate_pipeline(pending_texts, source_lang, target_lang)

        for idx, original_text, translated_text in zip(pending_indices, pending_texts, translated_pending):
            final_results[idx] = translated_text
            cache_key = (source_lang, target_lang, original_text)
            with TRANSLATION_CACHE_LOCK:
                TRANSLATION_CACHE[cache_key] = translated_text

    return final_results

def merge_and_group(results, scale):
    """
    Groups OCR results by their vertical position to reconstruct paragraphs.
    """
    items = []
    for r in results:
        y = r[0][0][1] * scale
        text = str(r[1]).strip()
        if len(text) > 0:
            items.append({"y": y, "text": text, "box": r[0]})
    
    items.sort(key=lambda x: x["y"])
    
    if not items: return []
    
    grouped = []
    current_group = items[0]
    
    for i in range(1, len(items)):
        if items[i]["y"] - current_group["y"] < 50:
            current_group["text"] += " " + items[i]["text"]
        else:
            grouped.append(current_group)
            current_group = items[i]
    grouped.append(current_group)
    
    return grouped

def process_image_bytes(image_bytes, source_lang, target_lang):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    result, _ = ocr(thresh) 
    
    if not result: 
        return []

    filtered_results = [r for r in result if len(str(r[1]).strip()) > 1]
    
    scale = 1.0 
    grouped_blocks = merge_and_group(filtered_results, scale)
    
    texts_to_translate = [b["text"] for b in grouped_blocks]
    translated_texts = translate_texts_with_cache(texts_to_translate, source_lang, target_lang)
    
    output = []
    for block, translated in zip(grouped_blocks, translated_texts):
        output.append({
            "box": block["box"],
            "sourceText": block["text"],
            "translated": translated
        })
    return output


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    state = {
        "sourceLang": DEFAULT_SOURCE_LANG,
        "targetLang": DEFAULT_TARGET_LANG,
    }

    try:
        while True:
            message = await ws.receive()

            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue

                if payload.get("type") == "language_pair":
                    try:
                        state["sourceLang"] = normalize_lang(payload.get("sourceLang", DEFAULT_SOURCE_LANG))
                        state["targetLang"] = normalize_lang(payload.get("targetLang", DEFAULT_TARGET_LANG))
                        await ws.send_json({
                            "type": "language_ack",
                            "sourceLang": state["sourceLang"],
                            "targetLang": state["targetLang"],
                        })
                    except Exception as e:
                        await ws.send_json({
                            "type": "error",
                            "message": str(e),
                        })
                continue

            if message.get("bytes") is not None:
                data = message["bytes"]
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(
                    None,
                    process_image_bytes,
                    data,
                    state["sourceLang"],
                    state["targetLang"],
                )
                await ws.send_json(output)

    except WebSocketDisconnect:
        return
    except Exception as e:
        print(f"Error: {e}")
        try:
            await ws.send_json([])
        except Exception:
            pass
    
@app.post("/test-image")
async def test_image(file: UploadFile = File(...)):
    data = await file.read()

    output = process_image_bytes(
        data,
        DEFAULT_SOURCE_LANG,
        DEFAULT_TARGET_LANG
    )
    
    return output