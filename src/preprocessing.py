# ============================================
# src/preprocessing.py — Funciones de preprocesamiento
# ============================================
# Este módulo contiene las funciones para preparar los datos
# tanto para entrenamiento como para predicción en la app.

import pandas as pd
import numpy as np
import joblib
import os


# --- Mapeos de variables categóricas ---
# Estos mapeos convierten texto a números para que el modelo los entienda.
# Son los mismos que se usan en el notebook de entrenamiento.

EDUCATION_MAP = {
    "Graduate": 1,
    "Not Graduate": 0
}

SELF_EMPLOYED_MAP = {
    "Yes": 1,
    "No": 0
}


def load_encoders(models_path="models/"):
    """
    Carga los encoders/scalers guardados durante el entrenamiento.
    Si no se usaron scalers (LightGBM no los necesita), retorna None.
    
    Parámetros:
        models_path (str): Ruta a la carpeta de modelos
    
    Retorna:
        dict: Diccionario con encoders cargados o None
    """
    scaler_path = os.path.join(models_path, "scaler.pkl")
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        return {"scaler": scaler}
    
    # LightGBM no necesita normalización, así que puede no existir scaler
    return None


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada del usuario para que el modelo
    pueda hacer la predicción.
    
    Este es el paso crítico: convierte los inputs del formulario
    de Streamlit al formato exacto que espera el modelo entrenado.
    
    Parámetros:
        data (dict): Diccionario con los datos del formulario.
            Ejemplo:
            {
                "no_of_dependents": 2,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 5000000,
                "loan_amount": 15000000,
                "loan_term": 12,
                "cibil_score": 700,
                "residential_assets_value": 7000000,
                "commercial_assets_value": 3000000,
                "luxury_assets_value": 5000000,
                "bank_asset_value": 4000000
            }
    
    Retorna:
        pd.DataFrame: DataFrame con una fila lista para predicción
    """
    # Crear DataFrame con una sola fila
    df = pd.DataFrame([data])
    
    # --- Codificar variables categóricas ---
    # Convertir "Graduate"/"Not Graduate" a 1/0
    if "education" in df.columns:
        df["education"] = df["education"].map(EDUCATION_MAP)
    
    # Convertir "Yes"/"No" a 1/0
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map(SELF_EMPLOYED_MAP)
    
    # --- Orden de columnas ---
    # El modelo espera las features en este orden exacto
    expected_columns = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ]
    
    # Reordenar columnas al orden esperado
    df = df[expected_columns]
    
    return df


def clean_dataset(filepath: str) -> pd.DataFrame:
    """
    Limpia el dataset original para entrenamiento.
    Esta función la usa Persona 1 en el notebook.
    
    Pasos de limpieza:
    1. Eliminar columna loan_id (no es feature predictiva)
    2. Manejar valores nulos
    3. Codificar variables categóricas
    4. Eliminar duplicados
    
    Parámetros:
        filepath (str): Ruta al archivo CSV original
    
    Retorna:
        pd.DataFrame: Dataset limpio listo para entrenamiento
    """
    # Cargar datos
    df = pd.read_csv(filepath)
    
    # 1. Eliminar loan_id (no aporta información predictiva)
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])
    
    # 2. Manejar valores nulos
    # Variables numéricas: imputar con la mediana
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Variables categóricas: imputar con la moda
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 3. Eliminar duplicados
    df = df.drop_duplicates()
    
    # 4. Codificar variables categóricas
    if "education" in df.columns:
        df["education"] = df["education"].map(EDUCATION_MAP)
    
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map(SELF_EMPLOYED_MAP)
    
    # Codificar target: loan_status
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].map({
            "Approved": 1,
            "Rejected": 0
        })
    
    # 5. Limpiar espacios en blanco en strings antes de codificar
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    
    return df
