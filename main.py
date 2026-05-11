import asyncio
import io
import json
import time
from typing import List

import numpy as np
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from PIL import Image
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

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "id": "Bahasa Indonesia",
}

def normalize_lang(lang: str) -> str:
    value = (lang or "").strip().lower()
    return value if value in SUPPORTED_LANGUAGES else DEFAULT_SOURCE_LANGUAGE

def batch_translate(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    if not texts:
        return []
    
    if source_lang == target_lang:
        return texts

    google_codes = {
        "en": "en",
        "ja": "ja",
        "zh": "zh-CN",
        "id": "id"
    }
    g_source = google_codes.get(source_lang, "auto")
    g_target = google_codes.get(target_lang, "id")

    translator = GoogleTranslator(g_source, g_target)
    
    # Batch texts using a unique delimiter to prevent Google rate limits
    delimiter = " | "
    chunk_size = 8
    final_translations = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        combined_text = delimiter.join(chunk)
        
        try:
            translated_combined = translator.translate(combined_text)
            translated_list = [t.strip() for t in translated_combined.split(delimiter)]
            
            if len(translated_list) != len(chunk):
                # Fallback for current chunk
                final_translations.extend([translator.translate(t) for t in chunk])
            else:
                final_translations.extend(translated_list)
        except Exception:
            final_translations.extend([""] * len(chunk))

    return final_translations

def process_image_bytes(image_bytes: bytes, source_lang: str, target_lang: str, include_metrics: bool = False):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    ocr_start = time.time()
    res = ocr(img_array)
    ocr_time = time.time() - ocr_start

    if res.boxes is None or res.txts is None:
        return {"data": [], "metrics": {}} if include_metrics else []

    boxes = []
    texts = []
    scores = []
    needs_translation_flags = []

    for box, text, score in zip(res.boxes, res.txts, res.scores):
        text = str(text).strip()

        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf！？。、,.!?]', '', text)
        
        try:
            score = float(score)
        except ValueError:
            score = 0.0

        # y_coords = [pt[1] for pt in box]
        # box_height = max(y_coords) - min(y_coords)

        if score > 0.40 and len(text) > 0:
            boxes.append([[float(c) for c in pt] for pt in box])
            texts.append(text)
            scores.append(score)

            if text.isnumeric() or (text.isascii() and len(text) <= 2):
                needs_translation_flags.append(False)
            else:
                needs_translation_flags.append(True)

    if not texts:
        return {"data": [], "metrics": {}} if include_metrics else []
    
    texts_to_translate = [t for t, flag in zip(texts, needs_translation_flags) if flag]

    trans_start = time.time()
    translated_subset = batch_translate(texts_to_translate, source_lang, target_lang)
    trans_time = time.time() - trans_start

    final_translations = []
    index = 0
    for text, flag in zip(texts, needs_translation_flags):
        if flag and index < len(translated_subset):
            final_translations.append(translated_subset[index])
            index += 1
        else:
            final_translations.append(text)

    output = [
        {
            "box": box,
            "sourceText": src_text,
            "translated": translated_text,
            "ocrScore": round(score, 4)
        }
        for box, src_text, translated_text, score in zip(boxes, texts, final_translations, scores)
        if translated_text # Drop empty translations
    ]

    if include_metrics:
        return {
            "data": output,
            "metrics": {
                "ocr_latency_sec": round(ocr_time, 4),
                "translation_latency_sec": round(trans_time, 4),
                "total_text_blocks": len(texts)
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
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(
                    None, process_image_bytes, message["bytes"], state["sourceLang"], state["targetLang"]
                )
                await ws.send_json(output)

    except WebSocketDisconnect:
        return
    
@app.post("/test-image")
async def test_image(file: UploadFile = File(...)):
    data = await file.read()
    output = process_image_bytes(
        data,
        DEFAULT_SOURCE_LANGUAGE,
        DEFAULT_TARGET_LANGUAGE,
        include_metrics=True
    )

    return output