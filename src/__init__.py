# ============================================
# src/__init__.py
# ============================================
# Hace que 'src' sea un paquete de Python y expone las
# funciones principales del proyecto de detección de fake news.

from .preprocessing import (
    limpiar_texto,
    cargar_y_combinar,
    crear_vectorizador,
    preprocess_input,
)
from .predict import load_model, make_prediction