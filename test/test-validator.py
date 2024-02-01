from guardrails import Guard
from pydantic import BaseModel, Field
from validator import ToxicLanguage


class ValidatorTestObject(BaseModel):
    test_val: str = Field(
        validators=[
            ToxicLanguage(on_fail="exception")
        ]
    )


TEST_OUTPUT = """
{
  "test_val": "What a lovely day it has been to roam around the city. I love it here!"
}
"""


guard = Guard.from_pydantic(output_class=ValidatorTestObject)

raw_output, guarded_output, *rest = guard.parse(TEST_OUTPUT)

print("validated output: ", guarded_output)


TEST_FAIL_OUTPUT = """
{
  "test_val": "What a lovely day it has been to roam around the city. I love it here! I hate how pathetic and needy you can be sometimes."
}
"""

try:
  guard.parse(TEST_FAIL_OUTPUT)
  print ("Failed to fail validation when it was supposed to")
except (Exception):
  print ('Successfully failed validation when it was supposed to')