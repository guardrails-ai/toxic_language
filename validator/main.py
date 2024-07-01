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
            on_fail=on_fail,
            threshold=threshold,
            validation_method=validation_method,
            **kwargs,
        )
        self._threshold = float(threshold)
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method
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
            results = self._model(value)
            if results:
                results = cast(List[List[Dict[str, Any]]], results)
                for label, score in results.items():
                    if label in self._labels and score > self._threshold:
                        pred_labels.append(label)
        return pred_labels

    def validate_each_sentence(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        sentences = nltk.sent_tokenize(value)

        toxic_sentences = []
        non_toxic_sentences = []

        for sentence in sentences:
            if sentence:
                pred_labels = self._inference(sentence)
                if pred_labels:
                    toxic_sentences.append(sentence)
                else:
                    non_toxic_sentences.append(sentence)

        if toxic_sentences:
            fixed_text = " ".join(non_toxic_sentences)
            error_spans = self.get_error_spans(value, fixed_text)
            toxic_sentences_text = "\n- ".join(toxic_sentences)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following sentences in your response "
                    "were found to be toxic:\n"
                    f"\n{toxic_sentences_text}"
                ),
                fix_value=fixed_text,
                error_spans=error_spans,
            )
        return PassResult(metadata=metadata)

    def validate_full_text(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        pred_labels = self._inference(value)
        if pred_labels:
            error_spans = [
                ErrorSpan(
                    start=0,
                    end=len(value),
                    reason=f"Toxic content detected: {', '.join(pred_labels)}",
                )
            ]
            return FailResult(
                metadata=metadata,
                error_message=(
                    "The generated text was found to be:\n" + ", ".join(pred_labels)
                ),
                fix_value="",
                error_spans=error_spans,
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

    def _inference_local(self, value: str) -> ValidationResult:
        """Local inference method for the toxic language validator."""
        return self.get_toxicity(value)

    def _inference_remote(self, value: str) -> ValidationResult:
        """Remote inference method for the toxic language validator."""
        request_body = {
            "text": value,
            "threshold": self._threshold,
        }
        request_body = json.dumps(request_body, ensure_ascii=False)
        response = self._hub_inference_request(request_body)
        return response["result"]

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
