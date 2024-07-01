import json
import torch
import nltk
from typing import Any, Dict, List
import detoxify

class InferlessPythonModel:

    def initialize(self):
        model_name = "unbiased-small"
        self.validation_method = "sentence"
        self.device = torch.device("cpu")
        self._model = detoxify.Detoxify(model_name, device=self.device)
        self._labels = [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexual_explicit",
        ]
        nltk.download('punkt')
        
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs["text"]
        threshold = inputs["threshold"]
        
        pred_labels = []
        if text:
            results = self._model.predict(text)
            if results:
                for label, score in results.items():
                    if label in self._labels and score > threshold:
                        pred_labels.append(label)
        
        return {"result": pred_labels}

    def finalize(self):
        pass
    
    
