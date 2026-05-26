# Predictor de Aprobación de Préstamos con LightGBM

Sistema web de Machine Learning que predice si una solicitud de préstamo bancario será **Aprobada** o **Rechazada** usando el algoritmo **LightGBM**.

La aplicación está **localizada para Colombia**: el usuario ingresa montos en **Pesos Colombianos (COP)** y el puntaje crediticio en la escala oficial de **Datacrédito (150–950)**. Internamente, los valores se convierten al espacio en el que fue entrenado el modelo (INR + CIBIL 300–900) y el resultado se devuelve con la probabilidad y el nivel de confianza.

---

## 🔗 Demostración

- **Aplicación desplegada:** [https://loan-approval-predictor.streamlit.app](https://loan-approval-predictor.streamlit.app)
- **Dataset:** [Loan Approval Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

---

## Problema que resuelve

La aprobación de préstamos es un proceso crítico en la industria financiera. Tradicionalmente, muchas solicitudes se evalúan de forma manual, lo que puede generar demoras, inconsistencias y decisiones poco trazables.

Este proyecto automatiza una evaluación inicial de crédito a partir de los datos del solicitante y las características del préstamo. El problema se aborda como una tarea de **clasificación binaria**:

- `Approved` (1): préstamo aprobado.
- `Rejected` (0): préstamo rechazado.

El caso de uso es fácil de demostrar: el usuario ingresa información financiera en la app y obtiene una respuesta inmediata.

---

## Solución propuesta

La solución integra:

1. **Dataset financiero:** registros históricos de solicitudes de préstamo (Kaggle).
2. **Preprocesamiento reproducible:** limpieza, codificación de categóricas y feature engineering centralizado en `src/preprocessing.py`. El mismo módulo se usa en el notebook de entrenamiento y en la app, garantizando que el modelo siempre reciba el formato exacto que vio en entrenamiento.
3. **Modelo LightGBM** entrenado en `notebooks/01_training_Grupo5.ipynb` y serializado a `models/modelo.pkl`.
4. **Aplicación Streamlit** (`app/app.py`) con UI personalizada (CSS/JS propios en `app/static/`).
5. **Localización Colombia:** conversión COP ↔ INR y Datacrédito ↔ CIBIL.
6. **Script de verificación** (`scripts/verify_model.py`) que comprueba accuracy ≥ 98% y valida el pipeline de extremo a extremo.
7. **Despliegue cloud** en Streamlit Cloud.

Flujo general:

```text
Usuario (COP + Datacrédito)
      ↓
Conversión (cop_to_inr, datacredito_to_cibil)
      ↓
Preprocesamiento (14 features = 11 originales + 3 derivadas)
      ↓
Modelo LightGBM (modelo.pkl)
      ↓
Resultado: Aprobado / Rechazado + Probabilidad + Confianza
```

---

## Algoritmo Utilizado

**LightGBM** (Light Gradient Boosting Machine) es un framework de gradient boosting desarrollado por Microsoft, basado en árboles de decisión.

Se eligió por:

- Alta eficiencia en datasets de tamaño mediano (4.269 registros).
- Manejo nativo de variables categóricas codificadas.
- Sin necesidad de normalización (modelos basados en árboles).
- Excelente balance entre velocidad de entrenamiento y precisión.
- Regularización incorporada (`reg_alpha`, `reg_lambda`) para prevenir sobreajuste.
- Soporte de importancia de features (clave para interpretabilidad financiera).
- Crecimiento *leaf-wise* + técnicas GOSS y EFB para máximo rendimiento.

### Hiperparámetros usados

```python
params = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      300,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "scale_pos_weight":  (y_train == 0).sum() / (y_train == 1).sum(),
    "random_state":      42,
}
```

Entrenamiento con `early_stopping(50)` sobre el split de test → **173 árboles entrenados** en ~0,1 s.

---

## Métricas de Desempeño

Métricas reales obtenidas en `notebooks/01_training_Grupo5.ipynb` (test set estratificado 80/20, 854 registros):

| Métrica          | Valor          |
| ---------------- | -------------- |
| **Accuracy**     | **99.77 %**    |
| Precision        | 99.62 %        |
| Recall           | 100.00 %       |
| F1-Score         | 99.81 %        |
| ROC-AUC          | 1.0000         |
| CV ROC-AUC (5-fold) | 1.0000 ± 0.0000 |

El modelo **supera ampliamente** el umbral mínimo del curso (Accuracy > 70 %, ROC-AUC > 0.8) y el objetivo del proyecto de **≥ 98 %** de aciertos sobre préstamos aprobados/rechazados.

> Para reproducir y verificar las métricas en tu máquina, ejecuta `python scripts/verify_model.py` (ver sección "Verificación del modelo").

---

## Dataset

| Característica       | Detalle                                         |
| -------------------- | ----------------------------------------------- |
| Fuente               | Kaggle                                          |
| Autor                | Archit Sharma                                   |
| Registros            | 4.269 solicitudes                               |
| Columnas             | 12 (11 predictivas + 1 target)                  |
| Variable objetivo    | `loan_status`                                   |
| Clases               | `Approved` (62,2 %) / `Rejected` (37,8 %)       |
| Tipo de problema     | Clasificación binaria                           |

Archivos incluidos en el repositorio:

```text
data/raw/loan_approval_dataset_RAW.csv
data/processed/loan_approval_clean_PROCESSED.csv
```

### Variables originales

| Variable                   | Tipo        | Descripción                                  |
| -------------------------- | ----------- | -------------------------------------------- |
| `no_of_dependents`         | Numérica    | Número de dependientes del solicitante       |
| `education`                | Categórica  | `Graduate` / `Not Graduate`                  |
| `self_employed`            | Categórica  | `Yes` / `No`                                 |
| `income_annum`             | Numérica    | Ingreso anual                                |
| `loan_amount`              | Numérica    | Monto del préstamo solicitado                |
| `loan_term`                | Numérica    | Plazo del préstamo (años)                    |
| `cibil_score`              | Numérica    | Puntaje crediticio (300–900)                 |
| `residential_assets_value` | Numérica    | Valor de activos residenciales               |
| `commercial_assets_value`  | Numérica    | Valor de activos comerciales                 |
| `luxury_assets_value`      | Numérica    | Valor de activos de lujo                     |
| `bank_asset_value`         | Numérica    | Valor de activos bancarios                   |
| **`loan_status`**          | **Target**  | **`Approved` / `Rejected`**                  |

### Feature engineering (3 features derivadas)

El módulo `src/preprocessing.py` añade automáticamente 3 features al pipeline. El modelo entrena y predice con **14 features en total**:

| Feature derivada  | Fórmula                                |
| ----------------- | -------------------------------------- |
| `debt_to_income`  | `loan_amount / (income_annum + 1)`     |
| `total_assets`    | suma de los 4 activos                  |
| `assets_to_loan`  | `total_assets / (loan_amount + 1)`     |

> El `+ 1` evita división por cero. Las mismas fórmulas se aplican en entrenamiento y en predicción en vivo, garantizando reproducibilidad.

---

## Preprocesamiento

El módulo `src/preprocessing.py` centraliza toda la preparación de datos:

```python
EDUCATION_MAP     = {"Graduate": 1, "Not Graduate": 0}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}
LOAN_STATUS_MAP   = {"Approved": 1, "Rejected": 0}
```

Orden exacto de las **14 columnas** que espera el modelo:

```text
no_of_dependents
education
self_employed
income_annum
loan_amount
loan_term
cibil_score
residential_assets_value
commercial_assets_value
luxury_assets_value
bank_asset_value
debt_to_income
total_assets
assets_to_loan
```

Funciones principales:

- `run_preprocessing_pipeline(raw_path, output_path)` — pipeline completo de entrenamiento (limpia, codifica, genera features, divide train/test estratificado 80/20).
- `preprocess_input(data: dict)` — convierte el formulario de la app en un DataFrame con las 14 columnas en el orden correcto.
- `clean_dataset(filepath)` — limpia el CSV crudo (duplicados, nulos, codificación) y agrega las 3 features derivadas.

---

## Localización Colombia

La app traduce las entradas del usuario al espacio en que vive el modelo:

| Entrada del usuario              | Conversión              | Valor que recibe el modelo  |
| -------------------------------- | ----------------------- | --------------------------- |
| Pesos Colombianos (COP)          | `÷ 50` (`COP_PER_INR`)  | INR (escala de entrenamiento) |
| Puntaje Datacrédito (150–950)    | Mapeo lineal            | CIBIL score (300–900)       |
| `"Profesional"` / `"No Profesional"` | Mapeo ES → EN       | `"Graduate"` / `"Not Graduate"` |
| `"Sí"` / `"No"`                  | Mapeo ES → EN           | `"Yes"` / `"No"`            |

Clasificación del puntaje Datacrédito que muestra la app:

- ≥ 781: Excelente
- 671–780: Bueno
- 561–670: Aceptable
- 320–560: Riesgo medio
- < 320: Alto riesgo

---

## Verificación del modelo

El script `scripts/verify_model.py` valida que el modelo entrenado:

1. Exista en disco y exponga `predict` / `predict_proba`.
2. Espere exactamente las 14 features definidas en `preprocessing.py`.
3. Alcance **≥ 98 % de accuracy** sobre el test set y sobre el dataset completo.
4. Funcione con el pipeline en vivo (`preprocess_input` + `make_prediction`) usando perfiles fuertes, débiles y los valores por defecto de la app.
5. Funcione con `batch_predict` sobre el split de test.

Ejecutar:

```bash
python scripts/verify_model.py
```

Salida esperada (resumen):

```text
[OK ] accuracy >= 98% (obtenido 0.9977)
[OK ] accuracy global >= 98% (obtenido 0.99XX)
MODELO VERIFICADO CORRECTAMENTE.
```

---

## Instalación Local

### Requisitos previos

- Python 3.11+ (probado en 3.13)
- pip
- Git

### Pasos

**Windows (PowerShell):**

```powershell
git clone https://github.com/XzJulian0711/loan-approval-predictor.git
cd loan-approval-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

**macOS / Linux:**

```bash
git clone https://github.com/XzJulian0711/loan-approval-predictor.git
cd loan-approval-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`.

### Reentrenar el modelo (opcional)

```bash
jupyter notebook notebooks/01_training_Grupo5.ipynb
```

Al ejecutar todas las celdas se regenera `models/modelo.pkl` y `models/feature_cols.json`.

---

## Uso de la Aplicación

1. Abre la app (local o desplegada).
2. Completa el formulario en sus 4 secciones:
   - **Datos del solicitante:** dependientes, educación, situación laboral.
   - **Información financiera:** ingreso anual (COP), monto del préstamo (COP), plazo en años.
   - **Historial crediticio:** puntaje Datacrédito (150–950).
   - **Valor de activos:** residenciales, comerciales, de lujo y bancarios (COP).
3. Haz clic en **"Predecir Aprobación del Préstamo"**.
4. Visualiza:
   - **Resultado**: Aprobado / Rechazado.
   - **Probabilidad** del modelo.
   - **Confianza** de la clasificación.
   - **Razón Préstamo / Ingreso** y total de activos.
5. Opcional: expande "Ver datos ingresados" para auditar los valores.

Ejemplo de salida:

```text
Resultado: Préstamo APROBADO
Probabilidad: 99.4 %
Confianza:    99.4 %
Razón Préstamo/Ingreso: 3.00x
```

---

## Estructura del Proyecto

```text
loan-approval-predictor/
├── README.md                                    # Este archivo
├── requirements.txt                             # Dependencias Python
├── .gitignore
├── .streamlit/
│   └── config.toml                              # Configuración Streamlit (static serving)
├── app/
│   ├── app.py                                   # Aplicación web Streamlit (Colombia)
│   └── static/
│       ├── css/styles.css                       # Estilos personalizados
│       ├── js/dashboard.js                      # Animaciones de métricas
│       └── img/logo.svg                         # Logo de la app
├── data/
│   ├── raw/
│   │   └── loan_approval_dataset_RAW.csv        # Dataset original
│   └── processed/
│       └── loan_approval_clean_PROCESSED.csv    # Dataset limpio con 14 features
├── models/
│   ├── modelo.pkl                               # LightGBM serializado
│   └── feature_cols.json                        # Orden de las 14 features
├── notebooks/
│   └── 01_training_Grupo5.ipynb                 # Entrenamiento, EDA y métricas
├── scripts/
│   └── verify_model.py                          # Verificación end-to-end (≥ 98 %)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                         # Limpieza + feature engineering
│   └── predict.py                               # Carga del modelo y predicción
└── docs/                                        # Imágenes EDA y diagramas
```

---

## Tecnologías

- Python 3.11+
- Pandas, NumPy
- Scikit-learn
- **LightGBM 4.x**
- Joblib (serialización)
- Matplotlib, Seaborn (EDA)
- **Streamlit** (frontend)
- GitHub + Streamlit Cloud (despliegue)

---

## Autores — Grupo 10

Inteligencia Artificial I · Fundación Universitaria Los Libertadores

| Nombre                            | Rol                          | Responsabilidades                                            |
| --------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| **Julian Camilo Cárdenas Torres** | Data & ML Engineer           | Dataset, EDA, entrenamiento LightGBM, serialización y verificación |
| **Jhon Alexander Vargas Catuche** | App Developer & DevOps       | Aplicación Streamlit, localización Colombia, despliegue cloud |
| **Juan Sebastián Vásquez Peña**   | Technical Writer & QA        | Documentación, presentación, validación de métricas          |

---

## Trabajo futuro

- Agregar explicabilidad con **SHAP** sobre cada predicción individual.
- Exponer un slider del umbral de decisión (actualmente 0.5) en la app.
- Añadir suite de pruebas (`pytest`) para `src/preprocessing.py` y `src/predict.py`.
- Calibrar probabilidades con `CalibratedClassifierCV`.
- Permitir cargar un CSV con múltiples solicitudes para predicción en lote desde la UI.

---

## Referencias

1. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye y T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *Advances in Neural Information Processing Systems*, vol. 30, 2017.
2. A. Sharma, "Loan Approval Prediction Dataset," Kaggle, 2023. [En línea]. Disponible en: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
3. Streamlit, "Streamlit Documentation," 2024. [En línea]. Disponible en: https://docs.streamlit.io/
4. Scikit-learn Developers, "Scikit-learn Documentation," 2024. [En línea]. Disponible en: https://scikit-learn.org/stable/
5. LightGBM Developers, "LightGBM Documentation." [En línea]. Disponible en: https://lightgbm.readthedocs.io/

---

## Licencia

Proyecto académico para la asignatura **Inteligencia Artificial I** — Actividad 3.
Distribuido bajo Licencia MIT si el equipo decide publicarlo en abierto.
