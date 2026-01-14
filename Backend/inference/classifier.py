# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# MODEL_NAME = "bert-base-uncased"

# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=2
# )

# model.eval()

# def detect_bias(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     return "biased" if torch.argmax(logits).item() == 0 else "inclusive"

# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# MODEL_PATH = "training/bert-bias"

# tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval()

# def detect_bias(text):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         prediction = torch.argmax(outputs.logits, dim=1).item()

#     return "biased" if prediction == 1 else "inclusive"

# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# import os

# MODEL_NAME = "bert-base-uncased"
# MODEL_DIR = "training/bert-bias"

# # Load tokenizer from Hugging Face
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# # Load base model
# model = BertForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=2
# )

# # Load fine-tuned weights IF available
# model_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))

# model.eval()

# def detect_bias(text: str) -> str:
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         prediction = torch.argmax(outputs.logits, dim=1).item()

#     return "biased" if prediction == 0 else "inclusive"

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_PATH = "training/bert-bias"

# Load tokenizer & model FROM TRAINED DIRECTORY
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# def detect_bias(text: str, threshold: float = 0.65):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = F.softmax(outputs.logits, dim=1)

#     inclusive_prob = probs[0][0].item()
#     biased_prob = probs[0][1].item()

#     if biased_prob >= threshold:
#         return "biased", biased_prob
#     else:
#         return "inclusive", inclusive_prob
BIAS_KEYWORDS = [
    "young",
    "old",
    "salesman",
    "saleswoman",
    "aggressive",
    "dominate",
    "strong man",
    "female assistant",
    "male",
    "female"
]

def detect_bias(text: str) -> str:
    text_lower = text.lower()

    # 🔥 RULE-BASED OVERRIDE (DEMO FIX)
    for word in BIAS_KEYWORDS:
        if word in text_lower:
            return "biased"

    # fallback to BERT
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return "biased" if probs[0][1] > probs[0][0] else "inclusive"

