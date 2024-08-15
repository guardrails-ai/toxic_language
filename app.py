from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import detoxify
import torch
import json

app = FastAPI()
# Initialize the detoxify model once
device = "cpu"
model = detoxify.Detoxify('unbiased-small', device=torch.device(device))

class InferenceData(BaseModel):
    name: str
    shape: List[int]
    data: Union[List[str], List[float]]
    datatype: str

class InputRequest(BaseModel):
    inputs: List[InferenceData]

class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]

@app.post("/validate", response_model=OutputResponse)
async def check_toxicity(input_request: InputRequest):
    text = None
    threshold = None
    
    for inp in input_request.inputs:
        if inp.name == "text":
            text = inp.data[0]
        elif inp.name == "threshold":
            threshold = float(inp.data[0])
    
    if text is None or threshold is None:
        raise HTTPException(status_code=400, detail="Invalid input format")
    results = model.predict(text)
    pred_labels = [label for label, score in results.items() if score > threshold]
    output_data = OutputResponse(
        modelname="unbiased-small",
        modelversion="1",
        outputs=[
            InferenceData(
                name="result",
                datatype="BYTES",
                shape=[len(pred_labels)],
                data=[pred_labels]
            )
        ]
    )
    
    return output_data
# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
