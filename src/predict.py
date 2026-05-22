# ============================================
# src/predict.py — Funciones de predicción
# ============================================
# Este módulo contiene las funciones para cargar el modelo
# entrenado y realizar predicciones.

import joblib
import pandas as pd
import numpy as np
import os


def load_model(model_path="models/modelo.pkl"):
    """
    Carga el modelo LightGBM entrenado desde el archivo .pkl.
    
    Parámetros:
        model_path (str): Ruta al archivo del modelo serializado
    
    Retorna:
        model: Modelo LightGBM cargado listo para predecir
    
    Raises:
        FileNotFoundError: Si el archivo del modelo no existe
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_path}'. "
            f"Asegúrate de haber ejecutado el notebook de entrenamiento primero."
        )
    
    model = joblib.load(model_path)
    print(f"✅ Modelo cargado correctamente desde: {model_path}")
    return model


def make_prediction(input_data: pd.DataFrame, model) -> dict:
    """
    Realiza una predicción usando el modelo LightGBM cargado.
    
    Parámetros:
        input_data (pd.DataFrame): DataFrame con los features preprocesados
                                    (una fila, 11 columnas)
        model: Modelo LightGBM cargado
    
    Retorna:
        dict: Diccionario con:
            - "prediction": 1 (Approved) o 0 (Rejected)
            - "label": "Approved" o "Rejected"
            - "probability": Probabilidad de aprobación (0.0 a 1.0)
            - "confidence": Porcentaje de confianza del modelo
    
    Ejemplo:
        >>> result = make_prediction(df_processed, model)
        >>> print(result)
        {
            "prediction": 1,
            "label": "Approved",
            "probability": 0.87,
            "confidence": 87.0
        }
    """
    try:
        # Obtener predicción (0 o 1)
        prediction = model.predict(input_data)[0]
        
        # Obtener probabilidades [prob_rejected, prob_approved]
        probabilities = model.predict_proba(input_data)[0]
        
        # Probabilidad de aprobación (clase 1)
        prob_approved = float(probabilities[1])
        
        # Construir resultado
        result = {
            "prediction": int(prediction),
            "label": "Aprobado ✅" if prediction == 1 else "Rechazado ❌",
            "probability": round(prob_approved, 4),
            "confidence": round(prob_approved * 100 if prediction == 1 
                               else (1 - prob_approved) * 100, 2)
        }
        
        return result
    
    except Exception as e:
        return {
            "prediction": -1,
            "label": f"Error: {str(e)}",
            "probability": 0.0,
            "confidence": 0.0
        }


def batch_predict(input_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Realiza predicciones en lote para múltiples registros.
    Útil para evaluar el modelo con el dataset de prueba.
    
    Parámetros:
        input_df (pd.DataFrame): DataFrame con múltiples filas
        model: Modelo LightGBM cargado
    
    Retorna:
        pd.DataFrame: DataFrame original + columnas de predicción
    """
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]
    
    result_df = input_df.copy()
    result_df["prediction"] = predictions
    result_df["probability_approved"] = probabilities
    result_df["label"] = result_df["prediction"].map({
        1: "Approved",
        0: "Rejected"
    })
    
    return result_df
