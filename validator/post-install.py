import nltk
import detoxify

# Download NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
print("NLTK stuff loaded successfully.")



model = detoxify.Detoxify("unbiased-small")
print("Detoxify's 'unbiased-small' toxicity model downloaded successfully!")
