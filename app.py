from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import detoxify
import torch
import os

from app_model import InputRequest, OutputResponse, ToxicLanguage

app = FastAPI()

toxic_language = ToxicLanguage()
toxic_language.load()

@app.post("/validate", response_model=OutputResponse)
async def check_toxicity(input_request: InputRequest):
    threshold = None
    for inp in input_request.inputs:
        if inp.name == "text":
            text_vals = inp.data
        elif inp.name == "threshold":
            threshold = float(inp.data[0])

    if text_vals is None or threshold is None:
        raise HTTPException(status_code=400, detail="Invalid input format")

    return toxic_language.infer(text_vals, threshold)


# Sagemaker specific endpoints
@app.get("/ping")
async def healtchcheck():
    return {"status": "ok"}

@app.post("/invocations", response_model=OutputResponse)
async def check_toxicity_sagemaker(input_request: InputRequest):
    return await check_toxicity(input_request)


# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
