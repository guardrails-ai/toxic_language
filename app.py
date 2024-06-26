import json
import torch
import nltk
from typing import Any, Dict, List
import detoxify

class InferlessPythonModel:

    def initialize(self):
        model_name = "unbiased-small"
        self.threshold = 0.5
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
        text = inputs["text"]  # Directly access the text input
        candidate_labels = inputs.get("candidate_labels", [])  # Directly access the candidate labels if needed
        
        if self.validation_method == "sentence":
            result = self.validate_each_sentence(text)
        elif self.validation_method == "full":
            result = self.validate_full_text(text)
        else:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        return result

    def get_toxicity(self, value: str) -> List[str]:
        """Check whether the generated text is toxic.

        Returns the labels predicted by the model with
        confidence higher than the threshold.

        Args:
            value (str): The generated text.

        Returns:
            pred_labels (bool): Labels predicted by the model
            with confidence higher than the threshold.
        """

        pred_labels = []
        if value:
            results = self._model.predict(value)
            if results:
                for label, score in results.items():
                    if label in self._labels and score > self.threshold:
                        pred_labels.append(label)
        return pred_labels

    def validate_each_sentence(self, value: str) -> Dict[str, Any]:
        """Validate that each sentence in the generated text is toxic."""

        sentences = nltk.sent_tokenize(value)

        unsupported_sentences, supported_sentences = [], []
        for sentence in sentences:
            if sentence:
                pred_labels = self.get_toxicity(sentence)
                if pred_labels:
                    unsupported_sentences.append(sentence)
                else:
                    supported_sentences.append(sentence)

        if unsupported_sentences:
            unsupported_sentences_text = "- " + "\n- ".join(unsupported_sentences)
            return {
                'generated_result': json.dumps({
                    "error_message": (
                        f"The following sentences in your response "
                        "were found to be toxic:\n"
                        f"\n{unsupported_sentences_text}"
                    ),
                    "fix_value": "\n".join(supported_sentences),
                    "metadata": {}
                })
            }
        return { "output" : json.dumps(value) } 

    def validate_full_text(self, value: str) -> Dict[str, Any]:
        """Validate that the entire generated text is toxic."""

        pred_labels = self.get_toxicity(value)
        if pred_labels:
            return {
                'generated_result': json.dumps({
                    "error_message": (
                        "The generated text was found to be:\n" + ",".join(pred_labels)
                    ),
                    "fix_value": "",
                    "metadata": {}
                })
            }
        return { "output" : json.dumps(value) } 

    def finalize(self):
        pass
