import pytest
from validator.main import ToxicLanguage
from guardrails import Guard

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception", use_local=True,
)
# Test passing response (non-toxic content)
def test_toxic_language_pass():
    response = guard.validate("Love how you think and attack the problem. Great job!")
    assert response.validation_passed is True

# Test failing response (toxic content)
def test_toxic_language_fail():
    with pytest.raises(Exception) as e:
        guard.validate(
            "Please look carefully. You are a stupid idiot who can't do anything right."
        )
    assert "Validation failed for field with errors:" in str(e.value)

# Test passing response for full text validation (non-toxic content with multiple sentences)
def test_toxic_language_full_text_pass():
    guard_full = Guard().use(
        ToxicLanguage, threshold=0.5, validation_method="full", on_fail="exception", use_local=True,
    )
    response = guard_full.validate(
        "This is a completely harmless and friendly message. It contains multiple sentences. "
        "All of them are non-toxic and should pass the validation."
    )
    assert response.validation_passed is True

# Test failing response for full text validation (toxic content with multiple sentences)
def test_toxic_language_full_text_fail():
    guard_full = Guard().use(
        ToxicLanguage, threshold=0.5, validation_method="full", on_fail="fix", use_local=True,
    )
    response = guard_full.validate(
        "This message starts innocently enough. However, you are a complete idiot. "
        "I hope bad things happen to you. The last sentence is also non-toxic."
    )
    print(f"Validation response: {response}")
    # assert response.validation_passed is False
    error_message = response.validation_summaries[0].failure_reason
    assert "The following text in your response was found to be toxic:" in error_message
    assert "This message starts innocently enough." in error_message
    assert "The last sentence is also non-toxic." in error_message
