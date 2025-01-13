import difflib
import json
from typing import Any, Callable, Dict, List, Optional, Union, cast

import detoxify
import nltk
import torch
from guardrails.validator_base import (
    ErrorSpan,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(
    name="guardrails/toxic_language", data_type="string", has_guardrails_endpoint=True
)
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

    This validator uses the pre-trained multi-label model from Detoxify -
    to check whether the generated text is toxic. If the model predicts any label 
    of: `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, 
    or `sexual_explicit` with confidence higher than the specified threshold, the validator 
    fails and returns the generated text with the toxic sentences / entire text removed. 
    Else the validator returns the generated text as it is.

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
            on_fail=on_fail,
            threshold=threshold,
            validation_method=validation_method,
            **kwargs,
        )
        self._threshold = float(threshold)
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method
        if self.use_local: 
            self._model = detoxify.Detoxify(model_name, device=torch.device(device)) #type: ignore
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
                for label, score in results.items(): # type: ignore
                    if label in self._labels and score > self._threshold:
                        pred_labels.append(label)
        return pred_labels

    def validate_each_sentence(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        sentences = nltk.sent_tokenize(value)

        unsupported_sentences, supported_sentences = [], []
        error_spans: List[ErrorSpan] = []
        char_index = 0

        sentence_predictions = self._inference(sentences)

        for idx, sentence in enumerate(sentences):
            pred_labels = sentence_predictions[idx]

            if pred_labels:
                unsupported_sentences.append(sentence)
                error_spans.append(
                    ErrorSpan(
                        start=char_index,
                        end=char_index + len(sentence),
                        reason=f"Toxic language detected: {', '.join(pred_labels)}",
                    )
                )
            else:
                supported_sentences.append(sentence)
            char_index += len(sentence) + 1  

        if unsupported_sentences:
            unsupported_sentences_text = "- " + "\n- ".join(unsupported_sentences)

            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following sentences in your response "
                    "were found to be toxic:\n"
                    f"\n{unsupported_sentences_text}"
                ),
                fix_value="\n".join(supported_sentences),
                error_spans=error_spans,
            )
        return PassResult(metadata=metadata)

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation method for the toxic language validator."""
        if not value:
            raise ValueError("Value cannot be empty.")
        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, metadata)
        else:
            return self.validate_full_text(value, metadata)

    def _inference_local(self, model_input: Union[str, list]) -> Any:
        """Local inference method for the toxic language validator."""

        if isinstance(model_input, str):
            model_input = [model_input]
        predictions = []
        for text in model_input:
            pred_labels = self.get_toxicity(text)
            predictions.append(pred_labels)
        
        return predictions

    def _inference_remote(self, model_input: Union[str, list]) -> Any:
        """Remote inference method for the toxic language validator."""

        if isinstance(model_input, str):
            model_input = [model_input]

        request_body = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [len(model_input)],
                    "data": model_input,
                    "datatype": "BYTES"
                },
                {
                    "name": "threshold",
                    "shape": [1],
                    "data": [self._threshold],
                    "datatype": "FP32"
                }
            ]
        }
        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)
        if not response or "outputs" not in response:
            raise ValueError("Invalid response from remote inference", response)

        data = [output["data"][0] for output in response["outputs"]]
        return data

    def get_error_spans(self, original: str, fixed: str) -> List[ErrorSpan]:
        """Generate error spans to display in failresult (if they exist). Error
        spans show the character level range of text that has failed validation.

        Args:
            original (str): The input string
            fixed (str): The 'validated' output string

        Returns:
            List[ErrorSpan]: A list of ErrorSpans to represent validation failures
            over the character sequence.
        """
        differ = difflib.Differ()
        diffs = list(differ.compare(original, fixed))
        error_spans = []
        if diffs:
            start = None
            end = None
            for i, diff in enumerate(diffs):
                if diff.startswith("- "):
                    if start is None:
                        start = i
                    end = i + 1
                else:
                    if start is not None and end is not None:
                        error_spans.append(
                            ErrorSpan(
                                start=start,
                                end=end,
                                reason=f"Toxic content detected in {start}:{end}",
                            )
                        )
                        start = None
                        end = None

            if start is not None and end is not None:
                error_spans.append(
                    ErrorSpan(
                        start=start,
                        end=end,
                        reason=f"Toxic content detected in {start}:{end}",
                    )
                )

        # Adjust indices to match the original string
        adjusted_spans = []
        original_index = 0
        diff_index = 0
        for span in error_spans:
            while diff_index < span.start:
                if not diffs[diff_index].startswith("+ "):
                    original_index += 1
                diff_index += 1
            start = original_index
            while diff_index < span.end:
                if not diffs[diff_index].startswith("+ "):
                    original_index += 1
                diff_index += 1
            adjusted_spans.append(
                ErrorSpan(
                    start=start,
                    end=original_index,
                    reason=span.reason.replace(str(span.start), str(start)).replace(
                        str(span.end), str(original_index)
                    ),
                )
            )

        return adjusted_spans

    def validate_full_text(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        pred_labels = self._inference([value])[0]

        if pred_labels:
            error_spans = [
                ErrorSpan(
                    start=0,
                    end=len(value),
                    reason=f"Toxic language detected: {', '.join(pred_labels)}",
                )
            ]

            return FailResult(
                metadata=metadata,
                error_message=(
                    "The following text in your response "
                    "was found to be toxic:\n"
                    f"\n{value}"
                ),
                fix_value="",
                error_spans=error_spans,
            )
        return PassResult(metadata=metadata)
