# ============================================
# src/predict.py — Funciones de predicción
# ============================================
# Carga el modelo (Regresión Logística) y el vectorizador TF-IDF, y
# clasifica una noticia (texto) como Verdadera o Falsa.
#
# Se cargan DOS archivos:
#   - modelo.pkl      (el clasificador lineal)
#   - vectorizer.pkl  (el TF-IDF que convierte texto en números)

import joblib
import os
import numpy as np


def load_model(model_path="models/modelo.pkl", vectorizer_path="models/vectorizer.pkl"):
    """
    Carga el modelo y el vectorizador desde disco.

    Devuelve una tupla (modelo, vectorizer). La app necesita los dos:
    el vectorizer para transformar la noticia nueva, el modelo para predecir.

    Raises
    ------
    FileNotFoundError
        Si falta alguno de los dos archivos.
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

    El modelo es una Regresión Logística sobre TF-IDF. En un modelo
    lineal, la contribución de cada palabra a la decisión es:

        contribución_i = coef_i * tfidf_i

    donde coef_i es el peso aprendido para esa palabra y tfidf_i su valor
    en este texto. El signo del coeficiente indica la dirección:
        - contribución > 0  -> empuja hacia FALSA (clase 1)
        - contribución < 0  -> empuja hacia VERDADERA (clase 0)

    Parámetros
    ----------
    X : matriz TF-IDF dispersa del texto (salida de preprocess_input).
    prediction : int, la clase que ganó (1 = Falsa, 0 = Verdadera).
    model : el clasificador lineal ya cargado (expone .coef_).
    vectorizer : el TF-IDF ya cargado (para mapear índices -> palabras).
    top_n : cuántas palabras influyentes devolver.

    Devuelve
    --------
    list[dict] con {"word": str, "weight": float} ordenada por influencia,
    solo de las palabras que empujaron hacia la clase predicha.
    """
    coef = np.asarray(model.coef_).ravel()  # un peso por palabra del vocabulario
    feature_names = vectorizer.get_feature_names_out()

    # X es disperso (1, n_features); recorremos solo las palabras presentes.
    Xc = X.tocoo()
    reasons = []
    for j, val in zip(Xc.col, Xc.data):
        weight = float(coef[j] * val)
        # Solo palabras que apoyan la clase ganadora:
        # si es Falsa (1) -> contribuciones positivas; si Verdadera (0) -> negativas.
        supports = weight > 0 if prediction == 1 else weight < 0
        if supports and abs(weight) > 1e-6:
            reasons.append({"word": feature_names[j], "weight": abs(weight)})

    reasons.sort(key=lambda r: -r["weight"])
    return reasons[:top_n]


def make_prediction(texto, model, vectorizer):
    """
    Clasifica una noticia como Verdadera o Falsa.

    Parámetros
    ----------
    texto : str
        La noticia (titular y/o cuerpo) escrita por el usuario.
    model : el clasificador lineal ya cargado.
    vectorizer : el TF-IDF ya cargado.

    Devuelve
    --------
    dict con:
        - prediction (int): 1 = Falsa, 0 = Verdadera
        - label (str): etiqueta legible
        - probability_fake (float): probabilidad de que sea falsa (0 a 1)
        - confidence (float): % de confianza en la clase predicha
        - reasons (list): palabras que más influyeron en la decisión
    """
    # Importamos aquí para evitar dependencias circulares
    from .preprocessing import preprocess_input

    try:
        # 1. Convertir el texto en números con el vectorizador
        X = preprocess_input(texto, vectorizer)

        # 2. Predecir clase y probabilidad
        prediction = int(model.predict(X)[0])
        prob_fake = float(model.predict_proba(X)[0][1])

        # 3. La confianza es la probabilidad de la clase que ganó
        confidence = prob_fake * 100 if prediction == 1 else (1 - prob_fake) * 100

        # 4. Explicar la decisión con las palabras más influyentes
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