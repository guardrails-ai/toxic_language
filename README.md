## Overview

| Developed by | Guardrails AI |
| Date of development | Feb 15, 2024 |
| Validator type | Moderation |
| Blog |  |
| License | Apache 2 |
| Input/Output | Output |

## Description

### Intended Use
This validator uses the pre-trained multi-label model from HuggingFace - `unitary/unbiased-toxic-roberta` to check whether the generated text is toxic. If the model predicts any label of: `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, or `sexual_explicit` with confidence higher than the specified threshold, the validator fails and returns the generated text with the toxic sentences / entire text removed. Else the validator returns the generated text as it is.

If validation_method is `sentence`, the validator will remove the sentences that are predicted to be toxic and return the remaining sentences. If validation_method is `full`, the validator will remove the entire text if the prediction is deemed toxic and return an empty string.

In our experiments, a threshold of 0.5 worked best, hence set as default here. However, you can try different values of threshold to see what works best for your use case. Link for experiments: https://wandb.ai/ml-guardrails/toxic-language-experiments

### Requirements

* Dependencies:
    - `nltk`
    - `transformers`
    - `torch`
    - guardrails-ai>=0.4.0

## Installation

```bash
$ guardrails hub install hub://guardrails/toxic_language
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails.hub import ToxicLanguage
from guardrails import Guard

# Use the Guard with the validator
guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)

# Test passing response
guard.validate("Love how you think and attack the problem. Great job!")

try:
    # Test failing response
    guard.validate(
        "Please look carefully. You are a stupid idiot who can't do anything right."
    )
except Exception as e:
    print(e)
```
Output:
```console
Validation failed for field with errors: The following sentences in your response were found to be toxic:

- You are a stupid idiot who can't do anything right.
```
**Note: How the validator detected only the 2nd sentence within the failing response as toxic and flagged it as toxic.**

# API Reference

**`__init__(self, threshold=0.5, validation_method="sentence", on_fail="noop")`**
<ul>

Initializes a new instance of the Validator class.

**Parameters:**

- **`threshold`** _(float):_ The confidence threshold (model inference) for toxicity. Defaults to 0.5.
- **`validation_method`** _(str):_ Whether to validate at the sentence level or over the full text. Must be one of `sentence` or `full`. Defaults to `sentence`
- **`on_fail`** *(str, Callable):* The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.

</ul>

<br>

**`__call__(self, value, metadata={}) -> ValidationResult`**

<ul>

Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters:**

- **`value`** *(Any):* The input value to validate.
- **`metadata`** *(dict):* A dictionary containing metadata required for validation. No additional metadata keys are needed for this validator.

</ul>
