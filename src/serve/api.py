import io
import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from src.models.registry import build_model
from src.data.schema import normalize_to_three_channels
from src.common.env import set_seed


LABELS = ["StartHesitation", "Turn", "Walking"]


class PredictResponse(BaseModel):
    label: str
    score: float
    probs: Optional[dict] = None


def load_checkpoint_model(checkpoint_path: str, model_name: str):
    model = build_model(model_name, num_classes=len(LABELS))
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    return model


def window_and_predict(df: pd.DataFrame, model: torch.nn.Module, sample_rate_hz: float = 100.0,
                       window_s: float = 5.0, overlap: float = 0.5) -> np.ndarray:
    # Normalize schema to 3-channel numeric array supporting both accelerometer and gait-parameter CSVs
    X = normalize_to_three_channels(df)  # (T, 3)
    # Standardize globally
    if X.size == 0:
        return np.zeros((len(LABELS),), dtype=np.float32)
    X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-6)
    step = int(window_s * sample_rate_hz * (1 - overlap))
    win = int(window_s * sample_rate_hz)
    if step <= 0:
        step = max(1, win // 2)
    starts = list(range(0, max(0, X.shape[0] - win + 1), step))
    if not starts:
        starts = [0]
    windows = []
    for s in starts:
        e = min(s + win, X.shape[0])
        w = X[s:e]
        if w.shape[0] < win:
            # pad at end
            pad = np.zeros((win - w.shape[0], X.shape[1]))
            w = np.concatenate([w, pad], axis=0)
        windows.append(w)
    xb = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)  # (B, T, C)
    with torch.no_grad():
        logits = model(xb)
        probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy()  # (B, K)
    return probs.mean(axis=0)  # average across windows


def create_app() -> FastAPI:
    set_seed(42)
    app = FastAPI(title="FOG Screening API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Config via env for simplicity
    ckpt = os.environ.get("FOG_CKPT", "artifacts/checkpoints/cnn_bilstm_fold0_best.pt")
    model_name = os.environ.get("FOG_MODEL", "cnn_bilstm")
    sample_rate = float(os.environ.get("FOG_SR", "100"))
    window_s = float(os.environ.get("FOG_WIN", "5"))
    overlap = float(os.environ.get("FOG_OVL", "0.5"))

    model = load_checkpoint_model(ckpt, model_name)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": model_name, "checkpoint": ckpt}

    @app.post("/api/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)):
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        probs = window_and_predict(df, model, sample_rate_hz=sample_rate, window_s=window_s, overlap=overlap)
        # Simple decision: abnormal if any event prob exceeds 0.5 threshold
        score = float(np.max(probs))
        label = "Abnormal (Flagged)" if score >= 0.5 else "Normal (Not flagged)"
        return PredictResponse(label=label, score=score, probs={LABELS[i]: float(probs[i]) for i in range(len(LABELS))})

    return app


app = create_app()


