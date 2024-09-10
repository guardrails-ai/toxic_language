from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import detoxify
import torch
import os

# Initialize the detoxify model once
env = os.environ.get("env", "dev")
torch_device = "cuda" if env == "prod" else "cpu"

print(f"Using torch device: {torch_device}")

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

class ToxicLanguage:

    model = None
    model_name = "unbiased-small"
    validation_method = "sentence"
    device = torch.device(torch_device)
    labels = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]

    _instance = None

    # Singleton pattern
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToxicLanguage, cls).__new__(cls)
        return cls._instance
       
    def load(self):
        self.model = detoxify.Detoxify("unbiased-small", device=torch.device(torch_device))

    def infer(self, text_vals, threshold) -> OutputResponse:
        outputs = []
        for idx, text in enumerate(text_vals):
            results = self.model.predict(text)
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