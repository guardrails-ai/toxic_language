from validator.main import ToxicLanguage

sample_text = "please look carefully. You are a stupid idiot who can't do anything right."

def create_remote_validator() -> ToxicLanguage:
    return ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        device="cpu",
        model_name="unbiased-small",
        on_fail=None,
        use_local=False  # Set to use remote inference
    )


remote_validator = create_remote_validator()
result = remote_validator._inference_remote(sample_text, {})
print(result)
assert 'toxicity' in result  # Ensure 'toxicity' is in the result