import pytest
from validator.main import ToxicLanguage
from guardrails import Guard

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
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
