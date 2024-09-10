from fastapi import FastAPI, HTTPException
from contextlib import contextmanager, asynccontextmanager
from pydantic import BaseModel
from typing import List
from app_inference_spec import InferenceSpec, InputRequest, OutputResponse


####################################################################
# FastAPI Setup & Endpoints
####################################################################

app = FastAPI()

inference_spec = InferenceSpec()

# Load the model once before the app starts 
# Not using lifespan events as they don't support sync functions.
@app.on_event("startup")
def startup_event():
    inference_spec.load()

@app.post("/validate", response_model=OutputResponse)
def validate(input_request: InputRequest):
    args, kwargs = inference_spec.process_request(input_request)
    return inference_spec.infer(*args, **kwargs)


####################################################################
# Sagemaker Specific Endpoints
####################################################################

@app.get("/ping")
async def healtchcheck():
    return {"status": "ok"}

@app.post("/invocations", response_model=OutputResponse)
def validate_sagemaker(input_request: InputRequest):
    args, kwargs = inference_spec.process_request(input_request)
    return inference_spec.infer(*args, **kwargs)


# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
