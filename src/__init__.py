# ============================================
# src/__init__.py — Loan Approval Predictor
# ============================================
# Este archivo hace que la carpeta src/ sea un paquete Python importable.

from .preprocessing import preprocess_input, load_encoders
from .predict import load_model, make_prediction
