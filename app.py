from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import detoxify
import torch
import os

app = FastAPI()
# Initialize the detoxify model once
env = os.environ.get("env", "dev")
torch_device = "cuda" if env == "prod" else "cpu"
model = detoxify.Detoxify("unbiased-small", device=torch.device(torch_device))


class InferenceData(BaseModel):
    name: str
    shape: List[int]
    data: List
    datatype: str


class InputRequest(BaseModel):
    inputs: List[InferenceData]


class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]


@app.get("/")
async def hello_world():
    return "toxic_language"

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

    return ToxicLanguage.infer(text_vals, threshold)


class ToxicLanguage:
    model_name = "unbiased-small"
    validation_method = "sentence"
    device = torch.device(torch_device)
    model = detoxify.Detoxify(model_name, device=device)
    labels = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]

    def infer(text_vals, threshold) -> OutputResponse:
        outputs = []
        for idx, text in enumerate(text_vals):
            results = ToxicLanguage.model.predict(text)
            pred_labels = [
                label for label, score in results.items() if score > threshold
            ]
            outputs.append(
                InferenceData(
                    name=f"result{idx}",
                    datatype="BYTES",
                    shape=[len(pred_labels)],
                    data=[pred_labels],
                )
            )

        output_data = OutputResponse(
            modelname="unbiased-small", modelversion="1", outputs=outputs
        )

        return output_data


# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
