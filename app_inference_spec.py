from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models_host.base_inference_spec import BaseInferenceSpec
from typing import List
import detoxify
import torch
import os

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

# Using same nomencalture as in Sagemaker classes
class InferenceSpec(BaseInferenceSpec):
    model = None

    model_name = "unbiased-small"
    validation_method = "sentence"
    labels = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]

    @property
    def torch_device(self):
        env = os.environ.get("env", "dev")
        torch_device = "cuda" if env == "prod" else "cpu"
        return torch_device
       
    def load(self):
        model_name = self.model_name
        torch_device = self.torch_device
        device = torch.device(torch_device)

        print(f"Loading model {model_name} using device {torch_device}...")
        self.model = detoxify.Detoxify(model_name, device=device)

    def process_request(self, input_request: InputRequest, mode="generic"):
        threshold = None
        for inp in input_request.inputs:
            if inp.name == "text":
                text_vals = inp.data
            elif inp.name == "threshold":
                threshold = float(inp.data[0])

        if text_vals is None or threshold is None:
            raise HTTPException(status_code=400, detail="Invalid input format")
        
        # args, and kwargs that are passed to the infer method
        args = (text_vals, threshold)
        kwargs = {}
        return args, kwargs

    # combined method of inference & process response 
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