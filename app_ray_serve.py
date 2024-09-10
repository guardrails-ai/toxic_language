from typing import Dict
from app_model import InputRequest, OutputResponse, ToxicLanguage
from fastapi import FastAPI, HTTPException

from ray import serve
from ray.serve.handle import DeploymentHandle

app = FastAPI()

@serve.deployment
class AppModel:
    def __init__(self):
        self._app_model = ToxicLanguage()
        self._app_model.load()

    def reconfigure(self, config: Dict):
        pass
        
    def infer(self, *args, **kwargs):
        return self._app_model.infer(*args, **kwargs)

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
@serve.ingress(app)
class Ingress:
    def __init__(self, app_model: DeploymentHandle):
        # Load model wrapper (app model)
        self.app_model = app_model

    def reconfigure(self, config: Dict):
        pass

    @app.post("/")
    async def check_toxicity(self, body: InputRequest):
        try:
            print("Received request")
            threshold = None
            for inp in body.inputs:
                if inp.name == "text":
                    text_vals = inp.data
                elif inp.name == "threshold":
                    threshold = float(inp.data[0])

            if text_vals is None or threshold is None:
                raise HTTPException(status_code=400, detail="Invalid input format")
            inference_result = await self.app_model.infer.remote(text_vals, threshold)
            print("Inference result: ", inference_result)
            return inference_result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
app = Ingress.bind(AppModel.bind())