from fastapi import FastAPI

import torch
import google.protobuf
import logging
import sentencepiece

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class TranslationItem(BaseModel):
    InputText: str

model_name = "SnypzZz/Llama2-13b-Language-translate"
try:
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    raise e

@app.post("/translate/")
async def translation_endpoint(item: TranslationItem):
    model_inputs = tokenizer(item.InputText, return_tensors="pt")

    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]
    )
    translated_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return {"translated_text": translated_output[0]}

'''
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
'''