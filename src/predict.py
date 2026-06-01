# ============================================
# src/predict.py — Funciones de predicción
# ============================================
# Carga el modelo LightGBM y el vectorizador TF-IDF, y clasifica
# una noticia (texto) como Verdadera o Falsa.
#
# Se cargan DOS archivos:
#   - modelo.pkl      (el clasificador LightGBM)
#   - vectorizer.pkl  (el TF-IDF que convierte texto en números)

import joblib
import os
import numpy as np
import pandas as pd


def load_model(model_path="models/modelo.pkl", vectorizer_path="models/vectorizer.pkl"):
    """
    Carga el modelo y el vectorizador desde disco.

    Devuelve una tupla (modelo, vectorizer). La app necesita los dos:
    el vectorizer para transformar la noticia nueva, el modelo para predecir.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_path}'. "
            f"Ejecuta primero el notebook 01_training_Grupo5.ipynb."
        )
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"No se encontró el vectorizador en '{vectorizer_path}'. "
            f"Ejecuta primero el notebook 01_training_Grupo5.ipynb."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"Modelo y vectorizador cargados correctamente.")
    return model, vectorizer


def explain_prediction(X, prediction, model, vectorizer, top_n=6):
    """
    Explica POR QUÉ el modelo tomó su decisión, listando las palabras
    de la noticia que más empujaron hacia la clase predicha.

    El modelo es LightGBM (boosting de árboles), no lineal, así que no
    tiene `coef_`. Para explicar usamos las contribuciones por feature
    de LightGBM (`pred_contrib`), que son equivalentes a valores SHAP:

        contribución_i = aporte de la palabra i a la log-odds de "Falsa"

    Signo:
        - contribución > 0  -> empuja hacia FALSA (clase 1)
        - contribución < 0  -> empuja hacia VERDADERA (clase 0)

    Parámetros
    ----------
    X : matriz TF-IDF dispersa del texto (salida de preprocess_input).
    prediction : int, la clase que ganó (1 = Falsa, 0 = Verdadera).
    model : el clasificador LightGBM ya cargado.
    vectorizer : el TF-IDF ya cargado (para mapear índices -> palabras).
    top_n : cuántas palabras influyentes devolver.

    Devuelve
    --------
    list[dict] con {"word": str, "weight": float} ordenada por influencia,
    solo de las palabras que empujaron hacia la clase predicha.
    """
    # LightGBM expone pred_contrib a través del booster interno.
    # Devuelve una fila con n_features + 1 valores: el último es el sesgo (bias).
    # Con entrada dispersa, LightGBM devuelve pred_contrib como matriz dispersa
    # (shape vacío al hacer np.asarray). Pasamos denso para obtener un ndarray.
    contribs = model.booster_.predict(X.toarray(), pred_contrib=True)[0]
    contribs = np.asarray(contribs)[:-1]  # quitamos el bias

    feature_names = vectorizer.get_feature_names_out()

    # X es dispersa (1, n_features); recorremos solo las palabras presentes en el texto.
    Xc = X.tocoo()
    reasons = []
    for j in Xc.col:
        weight = float(contribs[j])
        # Solo palabras que apoyan la clase ganadora:
        # si es Falsa (1) -> contribuciones positivas; si Verdadera (0) -> negativas.
        supports = weight > 0 if prediction == 1 else weight < 0
        if supports and abs(weight) > 1e-6:
            reasons.append({"word": feature_names[j], "weight": abs(weight)})

    reasons.sort(key=lambda r: -r["weight"])
    return reasons[:top_n]


def make_prediction(texto, model, vectorizer, umbral=0.5):
    """
    Clasifica una noticia como Verdadera o Falsa.

    Parámetros
    ----------
    texto : str
        La noticia ya preparada (titular y/o cuerpo según eligió el usuario).
    model : el clasificador LightGBM ya cargado.
    vectorizer : el TF-IDF ya cargado.
    umbral : float (0 a 1)
        Probabilidad mínima de "Falsa" para clasificarla como Falsa.
        Por defecto 0.5. Si el usuario lo sube (ej. 0.7), el modelo se
        vuelve más conservador: solo dice "Falsa" cuando está muy seguro.

    Devuelve
    --------
    dict con prediction, label, probability_fake, confidence, reasons.
    """
    from .preprocessing import preprocess_input

    try:
        # 1. Convertir el texto en números con el vectorizador
        X = preprocess_input(texto, vectorizer)

        # 2. Obtener la probabilidad de que sea Falsa
        prob_fake = float(model.predict_proba(X)[0][1])

        # 3. Aplicar el UMBRAL elegido por el usuario (input nuevo)
        #    En vez del 0.5 fijo de model.predict(), usamos el umbral.
        prediction = 1 if prob_fake >= umbral else 0

        # 4. La confianza es la probabilidad de la clase que ganó
        confidence = prob_fake * 100 if prediction == 1 else (1 - prob_fake) * 100

        # 5. Explicar la decisión con las palabras más influyentes
        reasons = explain_prediction(X, prediction, model, vectorizer)

        return {
            "prediction": prediction,
            "label": "Falsa" if prediction == 1 else "Verdadera",
            "probability_fake": round(prob_fake, 4),
            "confidence": round(confidence, 2),
            "reasons": reasons,
        }
    except Exception as e:
        return {
            "prediction": -1,
            "label": f"Error: {str(e)}",
            "probability_fake": 0.0,
            "confidence": 0.0,
            "reasons": [],
        }