# ============================================
# app/app.py — Aplicación Streamlit
# ============================================
# Dashboard para predicción de aprobación de préstamos en Colombia,
# usando un modelo LightGBM entrenado sobre el dataset público de Kaggle
# (Loan Approval Prediction, originalmente en rupias indias y score CIBIL).
#
# La interfaz está localizada para Colombia:
#   - Montos en Pesos Colombianos (COP)
#   - Historial crediticio basado en Datacrédito (150–950)
#
# Antes de predecir, convertimos los valores al espacio de entrenamiento
# del modelo (INR + CIBIL) para que las predicciones sean válidas.
#
# Estilos, scripts e imágenes viven en app/static/ y se sirven
# automáticamente vía `server.enableStaticServing = true`
# (configurado en .streamlit/config.toml).
#
# Ejecutar con: streamlit run app/app.py

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import sys
from pathlib import Path

# --- Agregar el directorio raíz al path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import preprocess_input
from src.predict import load_model, make_prediction

# ============================================
# RUTAS DE ASSETS ESTÁTICOS
# ============================================
APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
CSS_PATH = STATIC_DIR / "css" / "styles.css"
JS_PATH = STATIC_DIR / "js" / "dashboard.js"

LOGO_URL = "app/static/img/logo.svg"


# ============================================
# CONSTANTES DE LOCALIZACIÓN
# ============================================
# Tasa fija usada solo para mapear COP -> escala de entrenamiento (INR).
# No es una tasa de mercado en tiempo real; es un factor de conversión
# constante para que los valores que el usuario ingresa en pesos lleguen
# al modelo en el rango con el que fue entrenado.
COP_PER_INR = 50

# Rango oficial del puntaje Datacrédito en Colombia.
DATACREDITO_MIN = 150
DATACREDITO_MAX = 950

# Rango del puntaje CIBIL con el que fue entrenado el modelo.
CIBIL_MIN = 300
CIBIL_MAX = 900


def cop_to_inr(value_cop: float) -> float:
    """Convierte un monto en COP al rango monetario del entrenamiento."""
    return value_cop / COP_PER_INR


def datacredito_to_cibil(score: int) -> int:
    """Mapea linealmente Datacrédito (150–950) al rango CIBIL (300–900)."""
    pct = (score - DATACREDITO_MIN) / (DATACREDITO_MAX - DATACREDITO_MIN)
    return round(CIBIL_MIN + pct * (CIBIL_MAX - CIBIL_MIN))


def datacredito_tier(score: int):
    """
    Devuelve (clase_css, etiqueta) según los rangos comúnmente usados
    para clasificar el puntaje Datacrédito en Colombia.
    """
    if score >= 781:
        return ("excellent", "Excelente")
    if score >= 671:
        return ("good", "Bueno")
    if score >= 561:
        return ("regular", "Aceptable")
    if score >= 320:
        return ("low", "Riesgo Medio")
    return ("low", "Alto Riesgo")


def format_cop(value: float) -> str:
    """Formatea un número como pesos colombianos: $ 1.250.000"""
    return f"$ {int(round(value)):,}".replace(",", ".")


# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predictor de Préstamos Colombia",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
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
    possible_paths = [
        os.path.join(ROOT_DIR, "models", "modelo.pkl"),
        "models/modelo.pkl",
        "../models/modelo.pkl",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return load_model(path)
    st.error("No se encontró el modelo. Asegúrate de que 'models/modelo.pkl' existe.")
    st.stop()


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <img src="{LOGO_URL}" alt="Logo"/>
            <span>Predictor de Préstamos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Sobre el Proyecto")
    st.markdown(
        "Dashboard que predice si un préstamo será **aprobado o rechazado** "
        "para un solicitante en Colombia, mediante un modelo **LightGBM** "
        "entrenado sobre datos reales del sector financiero."
    )

    st.markdown("### Datos del Modelo")
    st.markdown(
        """
        - **Algoritmo:** LightGBM
        - **Registros de entrenamiento:** 4.269
        - **Variables predictivas:** 11
        - **Objetivo:** Estado del Préstamo
        - **Localización:** Pesos Colombianos (COP)
        - **Score crediticio:** Datacrédito
        """
    )

    st.markdown("### Equipo")
    st.markdown(
        """
        - **Persona 1** — Datos y Modelo
        - **Persona 2** — Aplicación y Despliegue
        - **Persona 3** — Documentación y Presentación
        """
    )

    st.markdown("### Enlaces")
    st.markdown(
        """
        - [Repositorio en GitHub](https://github.com/XzJulian0711/loan-approval-predictor)
        - [Conjunto de Datos en Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
        """
    )


# ============================================
# HEADER PRINCIPAL
# ============================================
st.markdown(
    f"""
    <div class="app-header">
        <div class="app-header__inner">
            <div class="app-header__logo">
                <img src="{LOGO_URL}" alt="Predictor de Préstamos"/>
            </div>
            <div class="app-header__text">
                <h1>Predictor de Aprobación de Préstamos</h1>
                <p>Ingresa los datos del solicitante en pesos colombianos y obtén una decisión instantánea respaldada por un modelo LightGBM. El historial crediticio se evalúa con el puntaje Datacrédito.</p>
            </div>
            <div class="app-header__badge">Modelo activo</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# CARGAR MODELO
# ============================================
model = get_model()


# ============================================
# FORMULARIO — Datos personales
# ============================================
st.markdown(
    """
    <div class="section-title">
        <span class="section-title__icon">1</span>
        Datos del Solicitante
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    no_of_dependents = st.number_input(
        "Número de Dependientes",
        min_value=0, max_value=10, value=2, step=1,
        help="Cantidad de personas que dependen económicamente del solicitante",
    )

with col2:
    education = st.selectbox(
        "Nivel de Educación",
        options=["Profesional", "No Profesional"],
        index=0,
        help="Si el solicitante tiene título universitario o no",
    )

with col3:
    self_employed = st.selectbox(
        "¿Trabaja por Cuenta Propia?",
        options=["No", "Sí"],
        index=0,
        help="Si el solicitante es independiente o empleado",
    )


# ============================================
# FORMULARIO — Información financiera (en COP)
# ============================================
st.markdown(
    """
    <div class="section-title">
        <span class="section-title__icon">2</span>
        Información Financiera
    </div>
    """,
    unsafe_allow_html=True,
)

col4, col5, col6 = st.columns(3)

with col4:
    income_annum_cop = st.number_input(
        "Ingreso Anual (COP)",
        min_value=10_000_000,
        max_value=500_000_000,
        value=250_000_000,
        step=5_000_000,
        format="%d",
        help="Ingreso anual total del solicitante en pesos colombianos",
    )
    st.caption(f"Equivale a {format_cop(income_annum_cop / 12)} / mes")

with col5:
    loan_amount_cop = st.number_input(
        "Monto del Préstamo (COP)",
        min_value=15_000_000,
        max_value=2_000_000_000,
        value=750_000_000,
        step=25_000_000,
        format="%d",
        help="Cantidad de dinero solicitada como préstamo en pesos colombianos",
    )
    st.caption(f"Solicita: {format_cop(loan_amount_cop)}")

with col6:
    loan_term = st.slider(
        "Plazo del Préstamo (años)",
        min_value=2, max_value=20, value=12, step=1,
        help="Duración total del préstamo en años",
    )


# ============================================
# FORMULARIO — Historial crediticio (Datacrédito)
# ============================================
st.markdown(
    """
    <div class="section-title">
        <span class="section-title__icon">3</span>
        Historial Crediticio (Datacrédito)
    </div>
    """,
    unsafe_allow_html=True,
)

datacredito_score = st.slider(
    "Puntaje Datacrédito",
    min_value=DATACREDITO_MIN,
    max_value=DATACREDITO_MAX,
    value=720,
    step=5,
    help=(
        "Puntaje crediticio del solicitante según Datacrédito Colombia. "
        "Rango oficial: 150 (alto riesgo) a 950 (excelente)."
    ),
)

tier, tier_label = datacredito_tier(datacredito_score)
gauge_pct = round(
    (datacredito_score - DATACREDITO_MIN) / (DATACREDITO_MAX - DATACREDITO_MIN) * 100,
    1,
)

st.markdown(
    f"""
    <div class="cibil-gauge">
        <div class="cibil-gauge__header">
            <span class="cibil-gauge__label">Puntaje Datacrédito</span>
            <span class="cibil-gauge__value">{datacredito_score}</span>
        </div>
        <div class="cibil-gauge__bar">
            <div class="cibil-gauge__fill cibil-gauge__fill--{tier}"
                 data-gauge-fill data-width="{gauge_pct}"
                 style="width: {gauge_pct}%"></div>
        </div>
        <span class="cibil-gauge__status cibil-gauge__status--{tier}">{tier_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# FORMULARIO — Valor de Activos (en COP)
# ============================================
st.markdown(
    """
    <div class="section-title">
        <span class="section-title__icon">4</span>
        Valor de Activos
    </div>
    """,
    unsafe_allow_html=True,
)

col7, col8 = st.columns(2)

with col7:
    residential_assets_cop = st.number_input(
        "Activos Residenciales (COP)",
        min_value=0, max_value=1_500_000_000, value=350_000_000,
        step=25_000_000, format="%d",
        help="Valor total de propiedades residenciales (casas, apartamentos)",
    )

with col8:
    commercial_assets_cop = st.number_input(
        "Activos Comerciales (COP)",
        min_value=0, max_value=1_000_000_000, value=150_000_000,
        step=25_000_000, format="%d",
        help="Valor total de propiedades comerciales (locales, bodegas, oficinas)",
    )

col9, col10 = st.columns(2)

with col9:
    luxury_assets_cop = st.number_input(
        "Activos de Lujo (COP)",
        min_value=0, max_value=1_250_000_000, value=250_000_000,
        step=25_000_000, format="%d",
        help="Valor de bienes de lujo (vehículos de alta gama, joyería, arte)",
    )

with col10:
    bank_asset_cop = st.number_input(
        "Activos Bancarios (COP)",
        min_value=0, max_value=750_000_000, value=200_000_000,
        step=25_000_000, format="%d",
        help="Valor total de cuentas, CDT e inversiones bancarias",
    )


# ============================================
# BOTÓN DE PREDICCIÓN
# ============================================
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button(
        "Predecir Aprobación del Préstamo",
        use_container_width=True,
        type="primary",
    )


# ============================================
# RESULTADO DE LA PREDICCIÓN
# ============================================
if predict_button:
    # 1) Convertir las entradas (COP + Datacrédito + ES) al espacio en
    #    que vive el modelo (INR + CIBIL + EN). El modelo nunca ve COP.
    education_en = "Graduate" if education == "Profesional" else "Not Graduate"
    self_employed_en = "Yes" if self_employed == "Sí" else "No"
    cibil_equivalent = datacredito_to_cibil(datacredito_score)

    model_input = {
        "no_of_dependents": no_of_dependents,
        "education": education_en,
        "self_employed": self_employed_en,
        "income_annum": cop_to_inr(income_annum_cop),
        "loan_amount": cop_to_inr(loan_amount_cop),
        "loan_term": loan_term,
        "cibil_score": cibil_equivalent,
        "residential_assets_value": cop_to_inr(residential_assets_cop),
        "commercial_assets_value": cop_to_inr(commercial_assets_cop),
        "luxury_assets_value": cop_to_inr(luxury_assets_cop),
        "bank_asset_value": cop_to_inr(bank_asset_cop),
    }

    try:
        processed_data = preprocess_input(model_input)
    except Exception as e:
        st.error(f"Error en preprocesamiento: {str(e)}")
        st.stop()

    result = make_prediction(processed_data, model)

    if result["prediction"] == 1:
        st.markdown(
            f"""
            <div class="result-card result-card--approved">
                <div class="result-card__icon">✓</div>
                <h2>Préstamo Aprobado</h2>
                <p>El modelo predice que este préstamo será aprobado.</p>
                <div class="result-card__confidence">Confianza · {result['confidence']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()
    elif result["prediction"] == 0:
        st.markdown(
            f"""
            <div class="result-card result-card--rejected">
                <div class="result-card__icon">✕</div>
                <h2>Préstamo Rechazado</h2>
                <p>El modelo predice que este préstamo será rechazado.</p>
                <div class="result-card__confidence">Confianza · {result['confidence']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error(f"Error en la predicción: {result['label']}")
        st.stop()

    # ----- Métricas detalladas en COP -----
    st.markdown(
        """
        <div class="section-title">
            <span class="section-title__icon">★</span>
            Detalles de la Predicción
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_assets_cop = (
        residential_assets_cop + commercial_assets_cop +
        luxury_assets_cop + bank_asset_cop
    )
    ratio = loan_amount_cop / income_annum_cop if income_annum_cop > 0 else 0
    prob_pct = result["probability"] * 100
    resultado_es = "Aprobado" if result["prediction"] == 1 else "Rechazado"

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 class="metric-card__label">Resultado</h3>
                <p class="metric-card__value">{resultado_es}</p>
                <div class="metric-card__sub">Clasificación final</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 class="metric-card__label">Probabilidad</h3>
                <p class="metric-card__value" data-counter
                   data-target="{prob_pct:.2f}" data-format="percent">0%</p>
                <div class="metric-card__sub">Score del modelo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 class="metric-card__label">Confianza</h3>
                <p class="metric-card__value" data-counter
                   data-target="{result['confidence']}" data-format="percent">0%</p>
                <div class="metric-card__sub">Certeza de la predicción</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 class="metric-card__label">Razón Préstamo / Ingreso</h3>
                <p class="metric-card__value" data-counter
                   data-target="{ratio:.2f}" data-format="ratio">0x</p>
                <div class="metric-card__sub">Total activos: {format_cop(total_assets_cop)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Ver datos ingresados", expanded=False):
        resumen = {
            "Dependientes": no_of_dependents,
            "Educación": education,
            "Trabaja por cuenta propia": self_employed,
            "Ingreso anual (COP)": format_cop(income_annum_cop),
            "Monto del préstamo (COP)": format_cop(loan_amount_cop),
            "Plazo (años)": loan_term,
            "Puntaje Datacrédito": datacredito_score,
            "Activos residenciales (COP)": format_cop(residential_assets_cop),
            "Activos comerciales (COP)": format_cop(commercial_assets_cop),
            "Activos de lujo (COP)": format_cop(luxury_assets_cop),
            "Activos bancarios (COP)": format_cop(bank_asset_cop),
            "Total activos (COP)": format_cop(total_assets_cop),
        }
        summary_df = pd.DataFrame.from_dict(resumen, orient="index", columns=["Valor"])
        summary_df.index.name = "Variable"
        st.dataframe(summary_df, use_container_width=True)


# ============================================
# FOOTER
# ============================================
st.markdown(
    """
    <div class="app-footer">
        <p class="app-footer__brand">Predictor de Préstamos Colombia</p>
        <p>Inteligencia Artificial I — Actividad 3 · Fundación Universitaria Los Libertadores</p>
        <p>Algoritmo: LightGBM · Localizado para Colombia (Pesos Colombianos · Datacrédito)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# INYECTAR JS — debe ir al final para que el DOM ya exista
# ============================================
inject_js(JS_PATH)
