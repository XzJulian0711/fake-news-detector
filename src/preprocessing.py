# ============================================
# src/preprocessing.py — Preprocesamiento de texto
# ============================================
# Limpia las noticias y las convierte a números con TF-IDF
# para que la Regresión Logística pueda clasificarlas como Falsas o Verdaderas.
#
# A diferencia del proyecto de préstamos (datos en columnas),
# aquí el input es TEXTO, así que el paso central es la
# vectorización TF-IDF.

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopwords en español: palabras tan comunes que no ayudan a distinguir
# (el, la, de, que...). Las quitamos para que el modelo se enfoque en
# las palabras que sí tienen significado.
STOPWORDS_ES = [
    "el", "la", "los", "las", "de", "del", "y", "a", "en", "que", "es",
    "un", "una", "unos", "unas", "por", "con", "para", "su", "sus", "se",
    "no", "lo", "al", "como", "mas", "mas", "o", "le", "les", "ya", "este",
    "esta", "estos", "estas", "ese", "esa", "esos", "esas", "fue", "han",
    "ha", "hay", "son", "ser", "fue", "muy", "sin", "sobre", "tambien",
    "me", "mi", "te", "tu", "nos", "pero", "si", "porque", "cuando", "donde",
]


def limpiar_texto(texto: str) -> str:
    """
    Limpia un texto crudo para el modelo:
    1. Lo pasa a minúsculas.
    2. Quita URLs (http...).
    3. Quita todo lo que no sean letras (números, signos, emojis).
    4. Quita espacios sobrantes.

    Devuelve el texto limpio, listo para vectorizar.
    """
    texto = str(texto).lower()
    texto = re.sub(r"http\S+", " ", texto)              # quitar enlaces
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)      # solo letras españolas
    texto = re.sub(r"\s+", " ", texto).strip()          # espacios sobrantes
    return texto


def cargar_y_combinar(ruta_train: str, ruta_dev: str) -> pd.DataFrame:
    """
    Carga los archivos train y development y los combina en un solo
    DataFrame. (El archivo 'test' del corpus no trae etiquetas claras,
    así que usamos train + development que sí están etiquetados.)

    Crea dos columnas nuevas:
    - 'target': 1 si la noticia es Falsa, 0 si es Verdadera.
    - 'texto': el titular + el cuerpo de la noticia, ya limpios.
    """
    tr = pd.read_excel(ruta_train)
    dv = pd.read_excel(ruta_dev)
    df = pd.concat([tr, dv], ignore_index=True)

    # Target binario: Fake -> 1, True -> 0
    df["target"] = (df["Category"].astype(str).str.strip().str.lower() == "fake").astype(int)

    # Combinar titular + texto (ambos aportan señales)
    df["texto"] = (df["Headline"].fillna("") + " " + df["Text"].fillna(""))
    df["texto"] = df["texto"].apply(limpiar_texto)

    # Quitar noticias que quedaron casi vacías tras limpiar
    df = df[df["texto"].str.len() > 10].reset_index(drop=True)

    return df


def crear_vectorizador() -> TfidfVectorizer:
    """
    Crea el vectorizador TF-IDF con la configuración del proyecto.

    - max_features=30000: vocabulario amplio (las combinaciones más útiles).
    - ngram_range=(1,2): considera palabras sueltas Y pares de palabras
      (ej: "agua milagrosa" como una sola señal).
    - min_df=3: ignora términos que aparecen en menos de 3 noticias (ruido).
    - sublinear_tf=True: usa 1+log(tf) en vez de tf crudo; suaviza el peso
      de palabras muy repetidas y mejora la clasificación de texto.
    - stop_words: quita las palabras vacías en español.

    OJO: este vectorizador se entrena (fit) en el notebook y se GUARDA
    como vectorizer.pkl, porque la app lo necesita para procesar texto nuevo.
    """
    return TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        stop_words=STOPWORDS_ES,
    )


def preprocess_input(texto: str, vectorizer: TfidfVectorizer):
    """
    Prepara UN texto escrito por el usuario en la app para predecir.

    1. Limpia el texto (misma limpieza que en entrenamiento).
    2. Lo transforma con el vectorizador YA entrenado.

    Devuelve la matriz numérica lista para model.predict().

    IMPORTANTE: usa .transform() (no .fit_transform()), porque el
    vectorizador ya fue entrenado; aquí solo aplicamos lo aprendido.
    """
    texto_limpio = limpiar_texto(texto)
    X = vectorizer.transform([texto_limpio])
    return X