"""
predict.py — Model inference module for TaGSense
Loads fine-tuned models from local disk.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODELS = {
    "mbert":             "models/taglish-mbert-sentiment",
    "xlm-roberta":       "models/taglish-xlm-roberta-sentiment",
    "tlunified-roberta": "models/taglish-tlunified-sentiment",
}

LABELS = ["Negative", "Neutral", "Positive"]

_model_cache = {}


def _load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]

    local_path = MODELS[model_name]
    print(f"[TaGSense] First request for {model_name} — loading from {local_path}...")

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_path,
        num_labels=3,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    _model_cache[model_name] = (tokenizer, model)
    print(f"[TaGSense] {model_name} ready and cached.")

    return tokenizer, model


def predict(text: str, model_name: str) -> dict:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

    tokenizer, model = _load_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

    label_id = int(probs.index(max(probs)))

    return {
        "label":    LABELS[label_id],
        "label_id": label_id,
        "confidence": {
            "negative": round(probs[0], 4),
            "neutral":  round(probs[1], 4),
            "positive": round(probs[2], 4),
        }
    }


def predict_all(text: str) -> dict:
    """Used by /compare — loads all 3 models, each lazy + cached."""
    return {name: predict(text, name) for name in MODELS}