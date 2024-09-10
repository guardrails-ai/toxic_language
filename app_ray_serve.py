from typing import Dict

from app_inference_spec import InputRequest, OutputResponse, InferenceSpec
from fastapi import FastAPI, HTTPException

from ray import serve
from ray.serve.handle import DeploymentHandle

app = FastAPI()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 1})
class InferenceDeployment:
    def __init__(self):
        self.spec = InferenceSpec()
        self.spec.load()

    def reconfigure(self, config: Dict):
        pass
        
    def infer(self, *args, **kwargs) -> OutputResponse:
        return self.spec.infer(*args, **kwargs)

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
@serve.ingress(app)
class Ingress:
    def __init__(self, inference_deployment: DeploymentHandle):
        self.spec = InferenceSpec()
        self.inference_deployment = inference_deployment

    def reconfigure(self, config: Dict):
        pass

    @app.post("/inference")
    async def validate(self, body: InputRequest) -> OutputResponse:
        try:
            args, kwargs = self.spec.process_request(body)
            # Infer call only goes to the GPU bound deployment, not the spec object direcly.
            # Note: we use async here although the inference is sync due to ray's async nature.
            inference_result = await self.inference_deployment.infer.remote(*args, **kwargs)
            return inference_result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

app = Ingress.bind(InferenceDeployment.bind())