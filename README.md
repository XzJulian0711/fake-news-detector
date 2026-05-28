# Detector de Noticias Falsas (Fake News) con LightGBM

Sistema web de Machine Learning que clasifica una noticia en español como **Falsa** o **Verdadera** usando **TF-IDF + LightGBM**, y explica **por qué** tomó la decisión mostrando las palabras que más influyeron.

El usuario pega el titular o el cuerpo de una noticia en la aplicación y obtiene, en una vista de dos columnas, el veredicto, el nivel de confianza y las palabras decisivas.

---

## Demostración

- **Aplicación desplegada:** _[(https://fake-news-detector-vmzgvo8jsoajvtozxgvh34.streamlit.app)](https://fake-news-detector-vmzgvo8jsoajvtozxgvh34.streamlit.app)_
- **Dataset:** corpus de noticias en español etiquetadas (`Fake` / `True`), incluido en `data/raw/`.

---

## Problema que resuelve

La desinformación se propaga más rápido que las noticias verificadas y su detección manual es lenta y costosa. Este proyecto automatiza una primera evaluación del texto de una noticia y la aborda como una tarea de **clasificación binaria**:

- `Falsa` (1): la noticia presenta patrones de desinformación.
- `Verdadera` (0): la noticia presenta lenguaje consistente con fuentes reales.

A diferencia de un clasificador "caja negra", la app también devuelve una **explicación**: las palabras del texto que más empujaron la decisión hacia falsa o verdadera.

### ¿Por qué elegimos la detección de noticias falsas?

- **Impacto social real:** la desinformación afecta procesos electorales, salud pública y la confianza en los medios. Es un problema vigente y medible, no un caso de estudio artificial.
- **Reto de NLP en español:** la mayoría de detectores y corpus disponibles están en inglés. Trabajar sobre noticias en español es un problema menos cubierto y más cercano a nuestro contexto.
- **Datos etiquetados y reproducibles:** existe un corpus académico abierto en español (`FakeNewsCorpusSpanish`), lo que permite entrenar, medir y comparar resultados de forma honesta.
- **Encaja con el alcance del curso:** es un problema de clasificación supervisada donde se aplica el flujo completo (limpieza → vectorización → modelo → evaluación → despliegue) sin necesitar infraestructura de gran escala.
- **Se presta a la explicabilidad:** el texto permite justificar la decisión palabra por palabra, lo que aporta valor pedagógico y de confianza frente a un modelo opaco.

### ¿Qué ofrece de diferente frente a otros detectores?

La mayoría de demos de detección de fake news se limitan a devolver una etiqueta (`Falsa`/`Verdadera`) sin contexto. Este proyecto se diferencia en:

- **Explicación palabra por palabra:** muestra las palabras que más empujaron la decisión (vía `pred_contrib` de LightGBM), no solo el veredicto. El usuario ve *por qué*, no solo *qué*.
- **Enfoque 100 % en español:** limpieza, *stopwords* y vocabulario TF-IDF pensados para el idioma, frente a la mayoría de herramientas centradas en inglés.
- **Honestidad sobre la confianza:** reporta el % de confianza real y advierte cuando el texto tiene pocas palabras reconocidas, en lugar de fingir certeza absoluta.
- **Reproducible y auditable de punta a punta:** corpus, notebook de entrenamiento, métricas y código de preprocesamiento compartido entre training y app están en el repositorio; cualquiera puede reentrenar y verificar.
- **Ligero y sin caja negra remota:** corre con un modelo local (TF-IDF + LightGBM), sin depender de APIs externas ni LLMs de pago para clasificar.

---

## Solución propuesta

1. **Corpus de noticias en español** etiquetadas como `Fake` / `True` (`data/raw/*.xlsx`).
2. **Preprocesamiento de texto reproducible** centralizado en `src/preprocessing.py`: limpieza (minúsculas, sin URLs, solo letras), eliminación de *stopwords* en español y vectorización **TF-IDF**. El mismo módulo se usa en el entrenamiento y en la app.
3. **Modelo LightGBM** entrenado en `notebooks/01_training_Grupo5_1.ipynb` y serializado a `models/modelo.pkl`. El vectorizador TF-IDF se guarda aparte en `models/vectorizer.pkl`.
4. **Aplicación Streamlit** (`app/app.py`) minimalista de dos columnas, con CSS/JS propios en `app/static/`.
5. **Explicabilidad**: `explain_prediction()` usa las contribuciones por característica de LightGBM (`pred_contrib`, equivalente a valores SHAP) para listar las palabras decisivas.
6. **Despliegue cloud** en Streamlit Cloud.

Flujo general:

```text
Usuario (texto de la noticia)
      ↓
Limpieza de texto (limpiar_texto)
      ↓
Vectorización TF-IDF (vectorizer.pkl)
      ↓
Modelo LightGBM (modelo.pkl)
      ↓
Resultado: Falsa / Verdadera + Confianza + Palabras decisivas (por qué)
```

---

## Algoritmo Utilizado

**LightGBM** (Light Gradient Boosting Machine), framework de gradient boosting de Microsoft basado en árboles de decisión, sobre una representación **TF-IDF** del texto.

Se eligió por:

- Eficiencia sobre matrices dispersas de alta dimensión (3.000 features TF-IDF).
- Buen balance entre velocidad de entrenamiento y precisión.
- Crecimiento *leaf-wise* + GOSS/EFB para máximo rendimiento.
- Soporte de contribuciones por feature (`pred_contrib`), clave para explicar la predicción palabra por palabra.

### Hiperparámetros usados

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    random_state=42,
    verbose=-1,
)
```

### Vectorizador TF-IDF

```python
TfidfVectorizer(
    max_features=3000,     # las 3000 palabras/combinaciones más útiles
    ngram_range=(1, 2),    # palabras sueltas y pares de palabras
    min_df=2,              # ignora términos que aparecen en una sola noticia
    stop_words=STOPWORDS_ES,
)
```

---

## Métricas de Desempeño

Métricas reales obtenidas en `notebooks/01_training_Grupo5_1.ipynb` sobre el conjunto de prueba (split estratificado 80/20, **195 noticias**):

| Métrica              | Valor             |
| -------------------- | ----------------- |
| **Accuracy**         | **77.44 %**       |
| Precision            | 77.66 %           |
| Recall               | 76.04 %           |
| F1-Score             | 76.84 %           |
| ROC-AUC              | 84.50 %           |
| CV ROC-AUC (5-fold)  | 0.8681 ± 0.0128   |

> Detección de texto en lenguaje natural: el desempeño es realista para un modelo TF-IDF + LightGBM sobre un corpus balanceado de ~1.000 noticias. No se espera la precisión casi perfecta de problemas tabulares.

---

## Dataset

| Característica       | Detalle                                              |
| -------------------- | ---------------------------------------------------- |
| Idioma               | Español                                              |
| Registros etiquetados| 971 noticias (train + development)                   |
| Clases               | `Verdadera` (491 · 50.6 %) / `Falsa` (480 · 49.4 %)  |
| Variable objetivo    | `Category` → `target` (Fake = 1, True = 0)           |
| Tipo de problema     | Clasificación binaria de texto                       |

### Fuente de los datos

Los datos provienen del corpus académico abierto **FakeNewsCorpusSpanish** (Posadas-Durán et al.), un conjunto de noticias en español etiquetadas como `Fake` / `True`. Los archivos se descargaron directamente de su repositorio oficial:

- Repositorio: https://github.com/jpposadas/FakeNewsCorpusSpanish
- Train: https://github.com/jpposadas/FakeNewsCorpusSpanish/raw/master/train.xlsx
- Development: https://github.com/jpposadas/FakeNewsCorpusSpanish/raw/master/development.xlsx
- Test: https://github.com/jpposadas/FakeNewsCorpusSpanish/raw/master/test.xlsx

### ¿Por qué usamos estos datasets?

- **En español y etiquetado a mano:** uno de los pocos corpus de fake news en español con etiquetas `Fake`/`True` verificadas, ideal para entrenamiento supervisado.
- **Balanceado:** ~50 % verdaderas y ~50 % falsas, lo que evita el sesgo hacia una clase y hace las métricas más confiables sin necesidad de remuestreo.
- **Multitema:** cubre varios temas (política, salud, ciencia, etc.), así el modelo no aprende un único dominio.
- **Split listo para usar:** ya viene separado en `train` y `development`, que combinamos para entrenar; `test` no trae etiqueta clara, por eso no se usa para medir.
- **Abierto y reproducible:** al ser público, cualquiera puede descargar los mismos `.xlsx` y reproducir nuestros resultados.

Archivos incluidos en el repositorio (copia local de las fuentes anteriores):

```text
data/raw/train.xlsx          # noticias de entrenamiento
data/raw/development.xlsx    # noticias de validación (se combinan con train)
data/raw/test.xlsx           # noticias sin etiqueta clara (no se usan para métricas)
```

### Columnas del corpus

| Columna     | Uso                                                        |
| ----------- | ---------------------------------------------------------- |
| `Category`  | Etiqueta `Fake` / `True` → se convierte en `target` (1/0)  |
| `Headline`  | Titular de la noticia → se combina en `texto`              |
| `Text`      | Cuerpo de la noticia → se combina en `texto`               |
| `Topic`, `Source`, `Link`, `Id` | Metadatos (no usados por el modelo)    |

`src/preprocessing.py::cargar_y_combinar()` combina `train + development`, crea `target` (Fake → 1) y `texto` (`Headline + Text` ya limpios), y descarta noticias que quedan casi vacías tras la limpieza.

---

## Preprocesamiento

El módulo `src/preprocessing.py` centraliza toda la preparación de texto:

- `limpiar_texto(texto)` — minúsculas, elimina URLs, deja solo letras (incluidas tildes y ñ), normaliza espacios.
- `cargar_y_combinar(train, dev)` — arma el DataFrame de entrenamiento con `target` y `texto`.
- `crear_vectorizador()` — construye el `TfidfVectorizer` con la configuración del proyecto.
- `preprocess_input(texto, vectorizer)` — prepara **un** texto del usuario: lo limpia y lo transforma con `.transform()` (el vectorizador ya está entrenado).

> El vectorizador se entrena (`fit`) en el notebook y se guarda como `models/vectorizer.pkl`. La app lo necesita para transformar texto nuevo al mismo espacio del entrenamiento.

---

## Explicabilidad — ¿por qué?

`src/predict.py::explain_prediction()` calcula **por qué** el modelo decidió, usando las contribuciones por feature de LightGBM (`booster_.predict(X, pred_contrib=True)`, equivalente a valores SHAP):

- Contribución **positiva** → la palabra empuja hacia **Falsa** (clase 1).
- Contribución **negativa** → la palabra empuja hacia **Verdadera** (clase 0).

La app toma las palabras presentes en el texto, se queda con las que apoyan la clase ganadora y muestra las **6 más influyentes** como chips de color. Si el texto tiene pocas palabras reconocidas por el vocabulario, la explicación es limitada y la app lo advierte.

`make_prediction()` devuelve:

```python
{
    "prediction": 1,            # 1 = Falsa, 0 = Verdadera
    "label": "Falsa",
    "probability_fake": 0.83,   # probabilidad de ser falsa (0–1)
    "confidence": 82.66,        # % de confianza en la clase ganadora
    "reasons": [{"word": "fuentes", "weight": 0.69}, ...],
}
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

La aplicación se abre en `http://localhost:8501`.

### Reentrenar el modelo (opcional)

```bash
jupyter notebook notebooks/01_training_Grupo5_1.ipynb
```

Al ejecutar todas las celdas se regeneran `models/modelo.pkl` y `models/vectorizer.pkl`.

---

## Uso de la Aplicación

1. Abre la app (local o desplegada).
2. Pega el titular o el cuerpo de una noticia en el cuadro de texto de la izquierda.
3. Haz clic en **"Analizar noticia"**.
4. En la columna derecha visualiza:
   - **Veredicto:** Noticia Falsa (rojo) / Noticia Verdadera (verde).
   - **Confianza** del modelo (barra animada).
   - **Por qué:** las palabras que más influyeron en la decisión.

---

## Estructura del Proyecto

```text
fake-news-detector/
├── README.md                                    # Este archivo
├── requirements.txt                             # Dependencias Python
├── .gitignore
├── .streamlit/
│   └── config.toml                              # Configuración Streamlit (static serving)
├── app/
│   ├── app.py                                   # Aplicación web Streamlit (fake news)
│   └── static/
│       ├── css/styles.css                       # Estilos minimalistas
│       └── js/dashboard.js                       # Animación de la barra de confianza
├── data/
│   └── raw/
│       ├── train.xlsx                           # Noticias de entrenamiento
│       ├── development.xlsx                     # Noticias de validación
│       └── test.xlsx                            # Noticias sin etiqueta clara
├── models/
│   ├── modelo.pkl                               # Clasificador LightGBM serializado
│   └── vectorizer.pkl                           # Vectorizador TF-IDF entrenado
├── notebooks/
│   └── 01_training_Grupo5_1.ipynb               # Entrenamiento, EDA y métricas
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                         # Limpieza de texto + TF-IDF
│   └── predict.py                               # Carga, predicción y explicación
└── docs/                                        # Imágenes EDA y matriz de confusión
```

---

## Tecnologías

- Python 3.11+
- Pandas, NumPy
- Scikit-learn (TF-IDF)
- **LightGBM 4.x**
- Joblib (serialización)
- openpyxl (lectura de `.xlsx`)
- Matplotlib, Seaborn (EDA)
- **Streamlit** (frontend)
- GitHub + Streamlit Cloud (despliegue)

---

## Autores — Grupo 10

Inteligencia Artificial · Fundación Universitaria Los Libertadores

| Nombre                              | Rol                    | Responsabilidades                                              |
| ----------------------------------- | ---------------------- | -------------------------------------------------------------- |
| **Juan Sebastian Vasques Peña**     | Data & ML Engineer     | Corpus, EDA, vectorización TF-IDF, entrenamiento LightGBM      |
| **Julian Camilo Cárdenas Torres**   | App Developer & DevOps | Aplicación Streamlit, explicabilidad, despliegue cloud         |
| **Jhon Alexander Vargas Catuche**   | Technical Writer & QA  | Documentación, presentación, validación de métricas            |

---

## Trabajo futuro

- Ampliar el corpus y reentrenar para mejorar el accuracy por encima del 77 %.
- Probar embeddings (Word2Vec / transformers) frente a TF-IDF.
- Mostrar resaltado de las palabras decisivas directamente sobre el texto.
- Exponer un slider del umbral de decisión (actualmente 0.5) en la app.
- Añadir suite de pruebas (`pytest`) para `src/preprocessing.py` y `src/predict.py`.

---

## Referencias

1. G. Ke, Q. Meng, T. Finley, et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *Advances in Neural Information Processing Systems*, vol. 30, 2017.
2. S. M. Lundberg y S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017. (Base teórica de `pred_contrib`.)
3. Streamlit, "Streamlit Documentation," 2024. https://docs.streamlit.io/
4. Scikit-learn Developers, "TfidfVectorizer Documentation." https://scikit-learn.org/stable/
5. LightGBM Developers, "LightGBM Documentation." https://lightgbm.readthedocs.io/
6. J. P. Posadas-Durán, H. Gómez-Adorno, G. Sidorov y J. J. M. Escobar, "Detection of fake news in a new corpus for the Spanish language," *Journal of Intelligent & Fuzzy Systems*, 2019. Corpus: https://github.com/jpposadas/FakeNewsCorpusSpanish

---

## Licencia

Proyecto académico para la asignatura **Inteligencia Artificial**.
Distribuido bajo Licencia MIT si el equipo decide publicarlo en abierto.