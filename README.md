# 🏦 Predictor de Aprobación de Préstamos

Sistema inteligente de predicción de aprobación de préstamos bancarios usando Machine Learning con el algoritmo **LightGBM**. La aplicación permite a un usuario ingresar los datos de un solicitante de crédito y obtener una predicción instantánea sobre si el préstamo será aprobado o rechazado, junto con la probabilidad y confianza del modelo.

## 🔗 Demostración

> **Aplicación desplegada:** [https://loan-approval-predictor.streamlit.app](https://loan-approval-predictor.streamlit.app)  
> **Repositorio GitHub:** [https://github.com/tu-usuario/loan-approval-predictor](https://github.com/tu-usuario/loan-approval-predictor)

## 📋 Descripción del Problema

La aprobación de préstamos bancarios es un proceso crítico en la industria financiera. Tradicionalmente, esta decisión se toma de forma manual por analistas de crédito, lo cual puede ser lento, inconsistente y propenso a sesgos humanos. Este proyecto automatiza el proceso de evaluación de solicitudes de préstamo mediante un modelo de Machine Learning que analiza 11 variables del solicitante para predecir si el crédito debe ser aprobado o rechazado.

El modelo aprende patrones de decisión a partir de datos históricos reales de solicitudes de préstamos, considerando factores como el ingreso anual, el monto solicitado, el historial crediticio (CIBIL score), los activos del solicitante y su perfil demográfico.

## 🤖 Algoritmo Utilizado

**LightGBM** (Light Gradient Boosting Machine) es un framework de gradient boosting desarrollado por Microsoft que utiliza algoritmos de aprendizaje basados en árboles de decisión. Se eligió este algoritmo porque:

- Es altamente eficiente con datasets de tamaño mediano como el nuestro (4,269 registros)
- Maneja nativamente variables categóricas sin necesidad de one-hot encoding extensivo
- No requiere normalización de datos (basado en árboles de decisión)
- Ofrece un excelente balance entre velocidad de entrenamiento y precisión predictiva
- Incluye regularización incorporada que previene el sobreajuste
- Proporciona importancia de features, lo cual es valioso para interpretabilidad en el sector financiero

### Métricas de Desempeño

| Métrica | Valor |
|---------|-------|
| Accuracy | XX.X% |
| Precision | XX.X% |
| Recall | XX.X% |
| F1-Score | XX.X% |
| ROC-AUC | XX.X% |

> *Las métricas se actualizarán tras el entrenamiento final del modelo.*

## 📊 Dataset

- **Fuente:** [Loan Approval Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Autor:** Archit Sharma
- **Registros:** 4,269 solicitudes de préstamo
- **Features:** 13 columnas (11 predictivas + 1 ID + 1 target)
- **Target:** `loan_status` (Approved / Rejected)

### Variables del Dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| no_of_dependents | Numérica | Número de dependientes del solicitante |
| education | Categórica | Graduate / Not Graduate |
| self_employed | Categórica | Yes / No |
| income_annum | Numérica | Ingreso anual del solicitante (USD) |
| loan_amount | Numérica | Monto del préstamo solicitado (USD) |
| loan_term | Numérica | Plazo del préstamo (meses) |
| cibil_score | Numérica | Puntaje crediticio (300-900) |
| residential_assets_value | Numérica | Valor de activos residenciales |
| commercial_assets_value | Numérica | Valor de activos comerciales |
| luxury_assets_value | Numérica | Valor de activos de lujo |
| bank_asset_value | Numérica | Valor de activos bancarios |
| **loan_status** | **Target** | **Approved / Rejected** |

## 🚀 Instalación Local

### Requisitos Previos

- Python 3.11 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/loan-approval-predictor.git
cd loan-approval-predictor

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app/app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## 📖 Uso de la Aplicación

1. Abre la aplicación en tu navegador (local o URL desplegada)
2. Completa el formulario con los datos del solicitante en las 4 secciones: datos personales, información financiera, historial crediticio y valor de activos
3. Haz clic en el botón **"Predecir Aprobación del Préstamo"**
4. Visualiza el resultado: aprobación o rechazo, probabilidad y confianza del modelo
5. Opcionalmente, expande la sección de datos ingresados para verificar la información

## 📁 Estructura del Proyecto

```
loan-approval-predictor/
├── README.md                  # Documentación principal
├── requirements.txt           # Dependencias Python
├── .gitignore                 # Archivos ignorados por Git
├── data/
│   ├── raw/                   # Dataset original sin modificar
│   └── processed/             # Dataset limpio y procesado
├── notebooks/
│   └── 01_training.ipynb      # Notebook de entrenamiento del modelo
├── models/
│   └── modelo.pkl             # Modelo LightGBM serializado
├── src/
│   ├── __init__.py            # Inicializador del paquete
│   ├── preprocessing.py       # Funciones de limpieza y preprocesamiento
│   └── predict.py             # Funciones de carga y predicción
├── app/
│   └── app.py                 # Aplicación web Streamlit
└── docs/
    └── presentacion.pdf       # Presentación final del proyecto
```

## 👥 Autores

| Nombre | Rol | Responsabilidades |
|--------|-----|-------------------|
| Persona 1 | Data & ML Engineer | Dataset, EDA, entrenamiento del modelo, serialización |
| Persona 2 | App Developer & DevOps | Aplicación Streamlit, despliegue en la nube, repositorio |
| Persona 3 | Technical Writer | README, documentación, presentación, diagramas |

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## 📚 Referencias

1. T. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," *Advances in Neural Information Processing Systems*, vol. 30, 2017.
2. A. Sharma, "Loan Approval Prediction Dataset," Kaggle, 2023. [En línea]. Disponible en: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
3. Streamlit Documentation, 2024. [En línea]. Disponible en: https://docs.streamlit.io/
4. Scikit-learn Documentation, 2024. [En línea]. Disponible en: https://scikit-learn.org/stable/
