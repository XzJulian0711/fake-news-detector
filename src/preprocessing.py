# ============================================
# src/preprocessing.py — Funciones de preprocesamiento
# ============================================
# Limpieza y preparación de datos para el dataset de aprobación
# de préstamos (Loan Approval), tanto para entrenamiento como
# para la predicción en vivo desde la app web.
#
# IMPORTANTE: El modelo LightGBM fue entrenado con 14 features:
#   las 11 originales + 3 derivadas (feature engineering):
#     - debt_to_income
#     - total_assets
#     - assets_to_loan
# Estas 3 features se generan aquí para alimentar al modelo con el
# formato EXACTO que vio durante el entrenamiento.

import pandas as pd
import numpy as np
import joblib
import os

# --- Mapeos de variables categóricas ---
EDUCATION_MAP = {"Graduate": 1, "Not Graduate": 0}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}

# --- Orden EXACTO de las 14 features que espera el modelo ---
FEATURE_COLS = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
    "debt_to_income", "total_assets", "assets_to_loan",
]

ASSET_COLS = [
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las 3 features derivadas con las MISMAS fórmulas del
    entrenamiento. El '+ 1' en denominadores evita división por cero.
        debt_to_income = loan_amount / (income_annum + 1)
        total_assets   = suma de los 4 activos
        assets_to_loan = total_assets / (loan_amount + 1)
    """
    df = df.copy()
    df["debt_to_income"] = df["loan_amount"] / (df["income_annum"] + 1)
    df["total_assets"] = df[ASSET_COLS].sum(axis=1)
    df["assets_to_loan"] = df["total_assets"] / (df["loan_amount"] + 1)
    return df


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocesa los datos del formulario de Streamlit para la predicción.
    1) DataFrame de una fila  2) codifica categóricas si son texto
    3) crea las 3 derivadas   4) reordena a las 14 columnas del modelo.
    """
    df = pd.DataFrame([data])

    if "education" in df.columns and df["education"].dtype == object:
        df["education"] = df["education"].map(EDUCATION_MAP)
    if "self_employed" in df.columns and df["self_employed"].dtype == object:
        df["self_employed"] = df["self_employed"].map(SELF_EMPLOYED_MAP)

    df = _add_engineered_features(df)
    df = df[FEATURE_COLS]

    # Asegurar tipos numéricos (LightGBM exige int/float/bool, no str)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_encoders(models_path="models/"):
    """Carga scaler si existe. LightGBM no lo requiere, así que suele ser None."""
    scaler_path = os.path.join(models_path, "scaler.pkl")
    if os.path.exists(scaler_path):
        return {"scaler": joblib.load(scaler_path)}
    return None


def clean_dataset(filepath: str) -> pd.DataFrame:
    """
    Limpia el dataset crudo y aplica el mismo feature engineering.
    Devuelve el dataset con 14 features + target codificado.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    if "education" in df.columns:
        df["education"] = df["education"].map(EDUCATION_MAP)
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map(SELF_EMPLOYED_MAP)
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    df = _add_engineered_features(df)
    return df


# ============================================
# Pipeline para el NOTEBOOK de entrenamiento
# ============================================
# Lo usa notebooks/01_training.ipynb. Reutiliza clean_dataset()
# (limpieza + las mismas 14 features) y añade el split train/test.

from sklearn.model_selection import train_test_split

TARGET_COL = "loan_status"


def run_preprocessing_pipeline(raw_path: str, output_path: str = None,
                               test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline completo para entrenamiento:
    limpia el CSV crudo, crea las 14 features, guarda el procesado
    (opcional) y devuelve el split estratificado.

    Garantiza que el notebook entrene con EXACTAMENTE las mismas
    features que la app genera en vivo (reproducibilidad).

    Retorna
    -------
    X_train, X_test, y_train, y_test
    """
    df = clean_dataset(raw_path)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[pipeline] Dataset procesado guardado en: {output_path}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[pipeline] Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {X.shape[1]}")
    return X_train, X_test, y_train, y_test
