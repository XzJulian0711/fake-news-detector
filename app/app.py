# ============================================
# app/app.py — Aplicación Streamlit
# ============================================
# Detector de Noticias Falsas (Fake News).
# El usuario pega una noticia (titular y/o cuerpo) y el modelo
# Regresión Logística + TF-IDF la clasifica como FALSA o VERDADERA, mostrando
# además POR QUÉ: las palabras que más influyeron en la decisión.
#
# Interfaz minimalista de dos columnas:
#   - Izquierda: el texto de la noticia + botón Analizar.
#   - Derecha: el veredicto, la confianza y la explicación.
#
# Estilos y scripts viven en app/static/ y se sirven vía
# `server.enableStaticServing = true` (en .streamlit/config.toml).
#
# Ejecutar con: streamlit run app/app.py

import streamlit as st
import streamlit.components.v1 as components
import os
import sys
from pathlib import Path

# --- Agregar el directorio raíz al path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.predict import load_model, make_prediction

# ============================================
# RUTAS DE ASSETS ESTÁTICOS
# ============================================
APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
CSS_PATH = STATIC_DIR / "css" / "styles.css"
JS_PATH = STATIC_DIR / "js" / "dashboard.js"


# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Detector de Noticias Falsas",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================
# HELPERS — Cargar assets externos
# ============================================
def inject_css(path: Path) -> None:
    """Inyecta un archivo CSS externo en la página."""
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_js(path: Path) -> None:
    """Inyecta un archivo JS externo. Se usa components.html porque
    los <script> dentro de st.markdown son sanitizados."""
    js = path.read_text(encoding="utf-8")
    components.html(f"<script>{js}</script>", height=0, width=0)


inject_css(CSS_PATH)


# ============================================
# CARGAR MODELO (cacheado)
# ============================================
@st.cache_resource
def get_model():
    model_path = os.path.join(ROOT_DIR, "models", "modelo.pkl")
    vectorizer_path = os.path.join(ROOT_DIR, "models", "vectorizer.pkl")
    try:
        return load_model(model_path, vectorizer_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


model, vectorizer = get_model()


# ============================================
# HEADER
# ============================================
st.markdown(
    """
    <div class="app-header">
        <h1>📰 Detector de Noticias Falsas</h1>
        <p>Pega el titular o el cuerpo de una noticia y descubre si es
        <strong>falsa</strong> o <strong>verdadera</strong>, y por qué.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# LAYOUT DE DOS COLUMNAS
# ============================================
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="panel-title">Noticia a analizar</div>', unsafe_allow_html=True)
    texto = st.text_area(
        "Texto de la noticia",
        height=320,
        placeholder="Pega aquí el titular y el cuerpo de la noticia...",
        label_visibility="collapsed",
    )
    analizar = st.button("Analizar noticia", type="primary", use_container_width=True)


with col_result:
    st.markdown('<div class="panel-title">Resultado</div>', unsafe_allow_html=True)

    if not analizar:
        st.markdown(
            """
            <div class="result-empty">
                <div class="result-empty__icon">🔍</div>
                <p>El veredicto aparecerá aquí.<br/>Escribe una noticia y pulsa <strong>Analizar</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not texto or not texto.strip():
        st.markdown(
            """
            <div class="result-empty result-empty--warn">
                <div class="result-empty__icon">✏️</div>
                <p>Escribe primero el texto de una noticia.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        result = make_prediction(texto, model, vectorizer)

        if result["prediction"] == -1:
            st.error(f"Error en la predicción: {result['label']}")
            st.stop()

        es_falsa = result["prediction"] == 1
        variant = "fake" if es_falsa else "real"
        icon = "✕" if es_falsa else "✓"
        titulo = "Noticia Falsa" if es_falsa else "Noticia Verdadera"
        confidence = result["confidence"]

        if es_falsa:
            descripcion = (
                "El modelo detectó patrones de lenguaje típicos de "
                "desinformación en este texto."
            )
            why_intro = "Estas palabras empujaron la decisión hacia <strong>falsa</strong>:"
        else:
            descripcion = (
                "El modelo encontró un lenguaje consistente con noticias "
                "reales y verificables."
            )
            why_intro = "Estas palabras empujaron la decisión hacia <strong>verdadera</strong>:"

        # ----- Tarjeta de veredicto -----
        st.markdown(
            f"""
            <div class="verdict verdict--{variant}">
                <div class="verdict__icon">{icon}</div>
                <div class="verdict__body">
                    <h2>{titulo}</h2>
                    <p>{descripcion}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----- Barra de confianza -----
        st.markdown(
            f"""
            <div class="confidence">
                <div class="confidence__head">
                    <span>Confianza del modelo</span>
                    <span class="confidence__value">{confidence:.1f}%</span>
                </div>
                <div class="confidence__bar">
                    <div class="confidence__fill confidence__fill--{variant}"
                         data-bar-fill data-width="{confidence:.1f}"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----- Explicación: por qué -----
        reasons = result.get("reasons", [])
        if reasons:
            chips = "".join(
                f'<span class="why-chip why-chip--{variant}">{r["word"]}</span>'
                for r in reasons
            )
            st.markdown(
                f"""
                <div class="why">
                    <p class="why__intro">{why_intro}</p>
                    <div class="why__chips">{chips}</div>
                    <p class="why__note">Las palabras con mayor peso según el
                    modelo (TF-IDF + Regresión Logística) en esta noticia.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="why why--empty">
                    <p>El texto tiene pocas palabras reconocidas por el modelo,
                    así que la explicación es limitada. Prueba con una noticia más larga.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )




# ============================================
# INYECTAR JS — al final para que el DOM ya exista
# ============================================
inject_js(JS_PATH)