from validator.main import ToxicLanguage
import pdb

# Sample text for testing
sample_text = "please look carefully. You are a stupid idiot who can't do anything right."

def create_local_validator() -> ToxicLanguage:
    return ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        device="cpu",
        model_name="unbiased-small",
        on_fail=None,
        use_local=True  # Set to use local inference
    )

def create_remote_validator() -> ToxicLanguage:
    return ToxicLanguage(
        threshold=0.5,
        validation_method="sentence",
        device="cpu",
        model_name="unbiased-small",
        on_fail=None,
        use_local=False  # Set to use remote inference
    )

def test_inference_local():
    """Test the local inference functionality."""
    local_validator = create_local_validator()
    result = local_validator._inference_local(sample_text, {})
    print(result)
    assert 'toxicity' in result  # Ensure 'toxicity' is in the result

def test_inference_remote():
    """Test the remote inference functionality."""
    pdb.set_trace()  # Debugger breakpoint
    remote_validator = create_remote_validator()
    result = remote_validator._inference_remote(sample_text, {})
    print(result)
    assert 'toxicity' in result  # Ensure 'toxicity' is in the result

if __name__ == "__main__":
    test_inference_local()
    test_inference_remote()
