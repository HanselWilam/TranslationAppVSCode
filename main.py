import asyncio
from fastapi import FastAPI, WebSocket
from rapidocr_onnxruntime import RapidOCR
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
import numpy as np
import io

app = FastAPI()
ocr = RapidOCR()

model_name = "Helsinki-NLP/opus-mt-ja-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

async def translate_async(text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, translate, text)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    while True:
        try:
            data = await ws.receive_bytes()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            img_array = np.array(image)

            result, _ = ocr(img_array)

            if not result:
                await ws.send_json([])
                continue

            boxes = [[[float(c) for c in pt] for pt in r[0]] for r in result]
            texts = [r[1] for r in result]

            translated_list = await asyncio.gather(*[translate_async(t) for t in texts])

            output = [
                {"box": box, "translated": translated}
                for box, translated in zip(boxes, translated_list)
            ]

            await ws.send_json(output)

        except Exception as e:
            print(f"Error: {e}")
            await ws.send_json([])