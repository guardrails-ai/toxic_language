from typing import Any, Callable, Dict, List, Optional, Union, cast

import nltk
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
import detoxify
import torch

@register_validator(name="guardrails/toxic_language", data_type="string")
class ToxicLanguage(Validator):
    """Validates that the generated text is toxic.

    **Key Properties**
    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/toxic_language`       |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        threshold: The confidence threshold (model inference) for toxicity.
            Defaults to 0.5.
        validation_method: Whether to validate at the sentence level or
            over the full text. Must be one of `sentence` or `full`.
            Defaults to `sentence`

    This validator uses the pre-trained multi-label model from HuggingFace -
    `unitary/unbiased-toxic-roberta` to check whether the generated text is toxic.
    If the model predicts any label of: `toxicity`, `severe_toxicity`,
    `obscene`, `threat`, `insult`, `identity_attack`, or `sexual_explicit` with
    confidence higher than the specified threshold, the validator fails and returns
    the generated text with the toxic sentences / entire text removed. Else the
    validator returns the generated text as it is.

    If validation_method is `sentence`, the validator will remove the sentences
    that are predicted to be toxic and return the remaining sentences. If
    validation_method is `full`, the validator will remove the entire text if
    the prediction is deemed toxic and return an empty string.

    In our experiments, a threshold of 0.5 worked best, hence set as default here.
    However, you can try different values of threshold to see what works best for
    your use case.
    Link for experiments: https://wandb.ai/ml-guardrails/toxic-language-experiments
    """

    def __init__(
        self,
        threshold: float = 0.5,
        validation_method: str = "sentence",
        device: Optional[Union[str, int]] = "cpu",
        model_name: Optional[str] = "unbiased-small", 
        on_fail: Union[Callable[..., Any], None] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail, threshold=threshold, validation_method=validation_method, **kwargs
        )
        
        self._threshold = float(threshold)
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method
        # Define the model, pipeline and labels
        self._model = detoxify.Detoxify(model_name, device=torch.device(device))
        self._labels = [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexual_explicit",
        ]

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

        # Get the model predictions and the list of labels
        # with confidence higher than the threshold
        pred_labels = []
        if value:
            results = self._model.predict(value)
            if results:
                results = cast(List[List[Dict[str, Any]]], results)
                for label, score in results.items():
                    if label in self._labels and score > self._threshold:
                        pred_labels.append(label)
        return pred_labels

    def validate_each_sentence(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that each sentence in the generated text is toxic."""

        # Split the value into sentences using nltk sentence tokenizer.
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
            unsupported_sentences = "- " + "\n- ".join(unsupported_sentences)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following sentences in your response "
                    "were found to be toxic:\n"
                    f"\n{unsupported_sentences}"
                ),
                fix_value="\n".join(supported_sentences),
            )
        return PassResult(metadata=metadata)

    def validate_full_text(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that the entire generated text is toxic."""

        pred_labels = self.get_toxicity(value)
        if pred_labels:
            return FailResult(
                metadata=metadata,
                error_message=(
                    "The generated text was found to be:\n" + ",".join(pred_labels)
                ),
                fix_value="",
            )
        return PassResult()

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation method for the toxic language validator."""
        if not value:
            raise ValueError("Value cannot be empty.")

        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, metadata)
        if self._validation_method == "full":
            return self.validate_full_text(value, metadata)
        raise ValueError("validation_method must be 'sentence' or 'full'.")
