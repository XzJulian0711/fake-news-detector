# ============================================
# src/predict.py — Funciones de predicción
# ============================================
# Carga el modelo LightGBM y el vectorizador TF-IDF, y clasifica
# una noticia (texto) como Verdadera o Falsa.
#
# A diferencia de préstamos, aquí se cargan DOS archivos:
#   - modelo.pkl      (el clasificador LightGBM)
#   - vectorizer.pkl  (el TF-IDF que convierte texto en números)

import joblib
import os


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


def make_prediction(texto, model, vectorizer):
    """
    Clasifica una noticia como Verdadera o Falsa.

    Parámetros
    ----------
    texto : str
        La noticia (titular y/o cuerpo) escrita por el usuario.
    model : el clasificador LightGBM ya cargado.
    vectorizer : el TF-IDF ya cargado.

    Devuelve
    --------
    dict con:
        - prediction (int): 1 = Falsa, 0 = Verdadera
        - label (str): etiqueta legible
        - probability_fake (float): probabilidad de que sea falsa (0 a 1)
        - confidence (float): % de confianza en la clase predicha
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

        return {
            "prediction": prediction,
            "label": "Falsa " if prediction == 1 else "Verdadera ",
            "probability_fake": round(prob_fake, 4),
            "confidence": round(confidence, 2),
        }
    except Exception as e:
        return {
            "prediction": -1,
            "label": f"Error: {str(e)}",
            "probability_fake": 0.0,
            "confidence": 0.0,
        }