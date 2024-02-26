import nltk
from transformers import pipeline

# Download NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
print("NLTK stuff loaded successfully.")

# Load pipeline once before actual initialization
# to avoid downloading during runtime
detoxify_pipeline = pipeline(
    "text-classification",
    model="unitary/unbiased-toxic-roberta",
    function_to_apply="sigmoid",
    top_k=None,
    padding="max_length",
    truncation=True,
)
print(f"Detoxify pipeline loaded successfully: {detoxify_pipeline}")
