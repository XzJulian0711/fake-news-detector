# ============================================
# app/app.py — Aplicación Streamlit
# ============================================
# Aplicación web para predicción de aprobación de préstamos
# usando un modelo LightGBM entrenado.
#
# Ejecutar con: streamlit run app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# --- Agregar el directorio raíz al path ---
# Esto permite importar los módulos de src/ sin problemas
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import preprocess_input
from src.predict import load_model, make_prediction

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predictor de Préstamos | LightGBM",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILOS CSS PERSONALIZADOS
# ============================================
st.markdown("""
<style>
    /* --- Header principal --- */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.05rem;
    }
    
    /* --- Tarjetas de resultado --- */
    .result-approved {
        background: linear-gradient(135deg, #064e3b, #065f46);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        border: 2px solid #10b981;
    }
    .result-rejected {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        border: 2px solid #ef4444;
    }
    .result-approved h2, .result-rejected h2 {
        color: white !important;
        font-size: 2rem;
        margin: 0;
    }
    .result-approved p, .result-rejected p {
        color: #d1d5db;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* --- Métricas --- */
    .metric-card {
        background: #1e293b;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-card h3 {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card p {
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    
    /* --- Info box --- */
    .info-box {
        background: #1e293b;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# CARGAR MODELO (se ejecuta una sola vez con cache)
# ============================================
@st.cache_resource
def get_model():
    """
    Carga el modelo LightGBM. Usa st.cache_resource para
    que solo se cargue una vez y no en cada interacción.
    """
    # Buscar el modelo en varias rutas posibles
    possible_paths = [
        os.path.join(ROOT_DIR, "models", "modelo.pkl"),
        "models/modelo.pkl",
        "../models/modelo.pkl",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return load_model(path)
    
    st.error("⚠️ No se encontró el modelo. Asegúrate de que 'models/modelo.pkl' existe.")
    st.stop()


# ============================================
# SIDEBAR — Información del proyecto
# ============================================
with st.sidebar:
    st.markdown("## 🏦 Sobre el Proyecto")
    st.markdown("""
    Esta aplicación predice si un préstamo será **aprobado o rechazado** 
    usando un modelo de Machine Learning **LightGBM**.
    """)
    
    st.divider()
    
    st.markdown("### 📊 Datos del Modelo")
    st.markdown("""
    - **Algoritmo:** LightGBM
    - **Dataset:** Loan Approval Prediction (Kaggle)
    - **Registros:** 4,269
    - **Features:** 11 variables predictivas
    - **Target:** loan_status (Approved/Rejected)
    """)
    
    st.divider()
    
    st.markdown("### 👥 Equipo")
    st.markdown("""
    - **Persona 1** — Data & Model
    - **Persona 2** — App & Deployment  
    - **Persona 3** — Docs & Presentation
    """)
    
    st.divider()
    
    st.markdown("### 🔗 Links")
    st.markdown("""
    - [📂 Repositorio GitHub](https://github.com/tu-usuario/loan-approval-predictor)
    - [📊 Dataset en Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
    """)


# ============================================
# HEADER PRINCIPAL
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🏦 Predictor de Aprobación de Préstamos</h1>
    <p>Ingresa los datos del solicitante y obtén una predicción instantánea usando inteligencia artificial</p>
</div>
""", unsafe_allow_html=True)


# ============================================
# CARGAR MODELO
# ============================================
model = get_model()


# ============================================
# FORMULARIO DE ENTRADA
# ============================================
st.markdown("### 📝 Datos del Solicitante")

# --- Fila 1: Datos personales ---
col1, col2, col3 = st.columns(3)

with col1:
    no_of_dependents = st.number_input(
        "👨‍👩‍👧‍👦 Número de Dependientes",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        help="Cantidad de personas que dependen económicamente del solicitante"
    )

with col2:
    education = st.selectbox(
        "🎓 Nivel de Educación",
        options=["Graduate", "Not Graduate"],
        index=0,
        help="Si el solicitante tiene título universitario o no"
    )

with col3:
    self_employed = st.selectbox(
        "💼 Trabajador Independiente",
        options=["No", "Yes"],
        index=0,
        help="Si el solicitante trabaja por cuenta propia"
    )

st.divider()

# --- Fila 2: Datos financieros ---
st.markdown("### 💰 Información Financiera")
col4, col5, col6 = st.columns(3)

with col4:
    income_annum = st.number_input(
        "📈 Ingreso Anual (USD)",
        min_value=200_000,
        max_value=10_000_000,
        value=5_000_000,
        step=100_000,
        format="%d",
        help="Ingreso anual total del solicitante"
    )

with col5:
    loan_amount = st.number_input(
        "💵 Monto del Préstamo (USD)",
        min_value=300_000,
        max_value=40_000_000,
        value=15_000_000,
        step=500_000,
        format="%d",
        help="Cantidad de dinero solicitada como préstamo"
    )

with col6:
    loan_term = st.slider(
        "📅 Plazo del Préstamo (meses)",
        min_value=2,
        max_value=20,
        value=12,
        step=1,
        help="Duración del préstamo en meses"
    )

# --- Fila 3: Puntaje crediticio ---
st.markdown("### 📋 Historial Crediticio")
cibil_score = st.slider(
    "🏆 Puntaje CIBIL (Credit Score)",
    min_value=300,
    max_value=900,
    value=650,
    step=10,
    help="Puntaje crediticio del solicitante. 300=Muy bajo, 900=Excelente"
)

# Mostrar indicador visual del puntaje
if cibil_score >= 750:
    st.success(f"Puntaje CIBIL: {cibil_score} — Excelente ⭐")
elif cibil_score >= 650:
    st.info(f"Puntaje CIBIL: {cibil_score} — Bueno 👍")
elif cibil_score >= 500:
    st.warning(f"Puntaje CIBIL: {cibil_score} — Regular ⚠️")
else:
    st.error(f"Puntaje CIBIL: {cibil_score} — Bajo 🔴")

st.divider()

# --- Fila 4: Activos ---
st.markdown("### 🏠 Valor de Activos")
col7, col8 = st.columns(2)

with col7:
    residential_assets_value = st.number_input(
        "🏠 Activos Residenciales (USD)",
        min_value=0,
        max_value=30_000_000,
        value=7_000_000,
        step=500_000,
        format="%d",
        help="Valor total de propiedades residenciales"
    )

with col8:
    commercial_assets_value = st.number_input(
        "🏢 Activos Comerciales (USD)",
        min_value=0,
        max_value=20_000_000,
        value=3_000_000,
        step=500_000,
        format="%d",
        help="Valor total de propiedades comerciales"
    )

col9, col10 = st.columns(2)

with col9:
    luxury_assets_value = st.number_input(
        "💎 Activos de Lujo (USD)",
        min_value=0,
        max_value=25_000_000,
        value=5_000_000,
        step=500_000,
        format="%d",
        help="Valor de bienes de lujo (vehículos, joyería, etc.)"
    )

with col10:
    bank_asset_value = st.number_input(
        "🏧 Activos Bancarios (USD)",
        min_value=0,
        max_value=15_000_000,
        value=4_000_000,
        step=500_000,
        format="%d",
        help="Valor total de cuentas y activos bancarios"
    )


# ============================================
# BOTÓN DE PREDICCIÓN
# ============================================
st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button(
        "🔮 Predecir Aprobación del Préstamo",
        use_container_width=True,
        type="primary"
    )

# ============================================
# RESULTADO DE LA PREDICCIÓN
# ============================================
if predict_button:
    
    # 1. Construir diccionario con los datos del formulario
    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    
    # 2. Preprocesar datos
    try:
        processed_data = preprocess_input(input_data)
    except Exception as e:
        st.error(f"Error en preprocesamiento: {str(e)}")
        st.stop()
    
    # 3. Hacer predicción
    result = make_prediction(processed_data, model)
    
    # 4. Mostrar resultado
    st.markdown("---")
    
    if result["prediction"] == 1:
        st.markdown(f"""
        <div class="result-approved">
            <h2>✅ Préstamo APROBADO</h2>
            <p>El modelo predice que este préstamo será aprobado con una confianza del {result['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    elif result["prediction"] == 0:
        st.markdown(f"""
        <div class="result-rejected">
            <h2>❌ Préstamo RECHAZADO</h2>
            <p>El modelo predice que este préstamo será rechazado con una confianza del {result['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Error en la predicción: {result['label']}")
        st.stop()
    
    # 5. Mostrar métricas detalladas
    st.markdown("### 📊 Detalles de la Predicción")
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Resultado</h3>
            <p>{result['label']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Probabilidad</h3>
            <p>{result['probability'] * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Confianza</h3>
            <p>{result['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        total_assets = (residential_assets_value + commercial_assets_value + 
                       luxury_assets_value + bank_asset_value)
        ratio = loan_amount / income_annum if income_annum > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Ratio Préstamo/Ingreso</h3>
            <p>{ratio:.1f}x</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 6. Resumen de datos ingresados
    with st.expander("📋 Ver datos ingresados", expanded=False):
        summary_df = pd.DataFrame([input_data]).T
        summary_df.columns = ["Valor"]
        summary_df.index.name = "Variable"
        st.dataframe(summary_df, use_container_width=True)


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <p>🎓 Inteligencia Artificial I — Actividad 3 | Fundación Universitaria Los Libertadores</p>
    <p>Algoritmo: LightGBM | Dataset: Loan Approval Prediction (Kaggle, 4,269 registros)</p>
</div>
""", unsafe_allow_html=True)
