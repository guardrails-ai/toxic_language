from http.client import HTTPException
from flask import Flask
import torch
import nltk
from typing import List, Union
import detoxify
from pydantic import BaseModel
from flask import request

app = Flask(__name__)

class InputData(BaseModel):
    name: str
    shape: List[int]
    data: Union[List[str], List[float]]
    datatype: str

class InputRequest(BaseModel):
    inputs: List[InputData]

class OutputData(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    data: List[Union[str, List[str]]]

class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[OutputData]

model_name = "unbiased-small"
validation_method = "sentence"
device = torch.device("cpu")
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
nltk.download('punkt')

@app.route("/")
def hello_world():
    return "Hello, World!"




@app.post("/validate")
def check_toxicity():
    input_request = request.json
    text = None
    threshold = None
    for inp in input_request['inputs']:
        if inp['name'] == "text":
            text_vals = inp['data']
        elif inp['name'] == "threshold":
            threshold = float(inp['data'][0])
    
    if text_vals is None or threshold is None:
        raise HTTPException(status_code=400, detail="Invalid input format")
    
    outputs = []
    for idx, text in enumerate(text_vals):
        results = model.predict(text)
        pred_labels = [label for label, score in results.items() if score > threshold]
        outputs.append(OutputData(
            name=f"result{idx}",
            datatype="BYTES",
            shape=[len(pred_labels)],
            data=[pred_labels]
        ))

    output_data = OutputResponse(
        modelname="unbiased-small",
        modelversion="1",
        outputs=outputs
    )
    
    return output_data.model_dump()


import bjoern

bjoern.run(app, "127.0.0.1", 8001)