"""
text_classifier.py
------------------
Inference-only text classification module.

Loads pre-trained artifacts:
  - DistilBERT + LoRA adapter (from ./distilbert_lora_toxic/)
  - ALBERT + LoRA adapter (from ./albert_lora_toxic/)
  - Bidirectional LSTM (from ./toxic_hybrid_artifacts/)
  - LogisticRegression meta-model (from ./toxic_hybrid_artifacts/)
"""

import re
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import joblib

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DISTIL_BASE = "distilbert-base-uncased"
ALBERT_BASE = "albert-base-v2"

DISTIL_DIR = Path("./distilbert_lora_toxic")
ALBERT_DIR = Path("./albert_lora_toxic")
ARTIFACT_DIR = Path("./toxic_hybrid_artifacts")


# ===========================================================================
# LSTM helpers (needed for inference)
# ===========================================================================

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\[\]sep]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ")


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 128) -> List[int]:
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab.get("<unk>", 1)) for tok in tokens][:max_len]
    if len(ids) < max_len:
        ids += [vocab.get("<pad>", 0)] * (max_len - len(ids))
    return ids


class ToxicLSTM(nn.Module):
    """Bidirectional LSTM for text classification."""

    def __init__(self, vocab_size, num_labels, emb_dim=128, hidden=128, dropout=0.3, bidir=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden, batch_first=True, bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        if self.lstm.bidirectional:
            last = torch.cat((h[-2], h[-1]), dim=1)
        else:
            last = h[-1]
        last = self.dropout(last)
        return self.fc(last)


# ===========================================================================
# Inference helpers
# ===========================================================================

@torch.no_grad()
def transformer_proba(model, tokenizer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


@torch.no_grad()
def lstm_proba(model, vocab, texts, max_len=128, batch_size=128) -> np.ndarray:
    model.eval()
    probs_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        x = torch.tensor(
            [encode_text(t, vocab, max_len=max_len) for t in batch], dtype=torch.long
        ).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_list.append(probs)
    return np.vstack(probs_list)


def build_meta_features(p_distil, p_albert, p_lstm) -> np.ndarray:
    return np.hstack([p_distil, p_albert, p_lstm])


# ===========================================================================
# PEFT model loader
# ===========================================================================

def load_peft_model(adapter_dir: str, base_model_name: str, num_labels: int):
    """Load a PEFT (LoRA) adapter on top of its base model."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# ===========================================================================
# TextClassifier — main public interface (inference only)
# ===========================================================================

class TextClassifier:
    """
    Hybrid ensemble text classifier (inference only).

    Loads model weights from the saved PEFT adapter folders and
    LSTM / meta-model artifacts. If artifacts are missing, raises
    a clear error.
    """

    def __init__(self):
        self.label_names: List[str] = []
        self.num_labels: int = 0
        self.distil_model = None
        self.distil_tok = None
        self.albert_model = None
        self.albert_tok = None
        self.lstm_model = None
        self.lstm_vocab: Dict[str, int] = {}
        self.meta = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load all pre-trained artifacts from disk."""

        # Check required files exist
        required = {
            "Label classes": ARTIFACT_DIR / "label_classes.json",
            "LSTM weights": ARTIFACT_DIR / "lstm.pt",
            "LSTM vocab": ARTIFACT_DIR / "lstm_vocab.json",
            "Meta-model": ARTIFACT_DIR / "meta_model.joblib",
            "DistilBERT adapter": DISTIL_DIR / "adapter_config.json",
            "ALBERT adapter": ALBERT_DIR / "adapter_config.json",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts: {', '.join(missing)}. "
                f"Please run 'python train.py' first to train and save all models."
            )

        print("[TextClassifier] Loading pre-saved artifacts …")

        # Label names
        self.label_names = json.loads(
            (ARTIFACT_DIR / "label_classes.json").read_text(encoding="utf-8")
        )
        self.num_labels = len(self.label_names)

        # DistilBERT + LoRA adapter
        self.distil_model, self.distil_tok = load_peft_model(
            str(DISTIL_DIR), DISTIL_BASE, self.num_labels
        )

        # ALBERT + LoRA adapter
        self.albert_model, self.albert_tok = load_peft_model(
            str(ALBERT_DIR), ALBERT_BASE, self.num_labels
        )

        # LSTM
        self.lstm_vocab = json.loads(
            (ARTIFACT_DIR / "lstm_vocab.json").read_text(encoding="utf-8")
        )
        self.lstm_model = ToxicLSTM(
            vocab_size=len(self.lstm_vocab), num_labels=self.num_labels
        ).to(DEVICE)
        self.lstm_model.load_state_dict(
            torch.load(ARTIFACT_DIR / "lstm.pt", map_location=DEVICE, weights_only=True)
        )
        self.lstm_model.eval()

        # Meta-model
        self.meta = joblib.load(ARTIFACT_DIR / "meta_model.joblib")

        print("[TextClassifier] All models loaded successfully.")

    def classify(self, query: str, image_caption: str = "") -> Dict:
        """
        Classify text using the hybrid ensemble.

        Returns
        -------
        dict : {"label": str, "confidence": float, "probs": {label: float, ...}}
        """
        text = (str(query).strip() + " [SEP] " + str(image_caption).strip()).strip()

        p_d = transformer_proba(self.distil_model, self.distil_tok, [text])[0]
        p_a = transformer_proba(self.albert_model, self.albert_tok, [text])[0]
        p_l = lstm_proba(self.lstm_model, self.lstm_vocab, [text])[0]

        X = build_meta_features(p_d.reshape(1, -1), p_a.reshape(1, -1), p_l.reshape(1, -1))
        probs = self.meta.predict_proba(X)[0]
        pred_id = int(np.argmax(probs))
        conf = float(np.max(probs))

        return {
            "label": self.label_names[pred_id],
            "confidence": conf,
            "probs": {
                self.label_names[i]: float(probs[i]) for i in range(self.num_labels)
            },
        }
