from guardrails import Guard
from pydantic import BaseModel, Field
from validator import ToxicLanguage
import pytest


# Create a pydantic model with a field that uses the custom validator
class ValidatorTestObject(BaseModel):
    text: str = Field(
        validators=[
            ToxicLanguage(
                threshold=0.5, validation_method="sentence", on_fail="exception"
            )
        ]
    )


# Test happy path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
            "text": "What a lovely day it has been to roam around the city. I love it here!"
        }
        """,
        """
        {
            "text": "The ice cream was so good, I could eat it all day!"
        }
        """,
    ],
)
def test_happy_path(value):
    # Create a guard from the pydantic model
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)
    response = guard.parse(value)
    print("Happy path response", response)
    assert response.validation_passed is True


# Test fail path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
            "text": "What a lovely day it has been to roam around the city. I love it here! I hate how pathetic and needy you can be sometimes."
        }
        """,
        """
        {
            "text": "Shut the hell up, you're so annoying. I can't stand you even in hell. See you tomorrow!"
        }
        """,
    ],
)
def test_fail_path(value):
    # Create a guard from the pydantic model
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)

    with pytest.raises(Exception):
        response = guard.parse(value)
        print("Fail path response", response)
