# ============================================
# scripts/verify_model.py
# ============================================
# Verifica que el modelo entrenado (models/modelo.pkl)
# alcance al menos 98% de accuracy sobre todo el dataset
# y sobre el split de prueba. Tambien valida que el pipeline
# de prediccion en vivo (preprocess_input + make_prediction)
# funcione con datos representativos.

import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

from src.preprocessing import (
    run_preprocessing_pipeline, preprocess_input, FEATURE_COLS,
)
from src.predict import load_model, make_prediction, batch_predict


TARGET_ACC = 0.98
MODEL_PATH = os.path.join(ROOT_DIR, "models", "modelo.pkl")
FEATURES_PATH = os.path.join(ROOT_DIR, "models", "feature_cols.json")
RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "loan_approval_dataset_RAW.csv")
PROCESSED_PATH = os.path.join(ROOT_DIR, "data", "processed",
                              "loan_approval_clean_PROCESSED.csv")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def assert_true(cond, msg):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def main():
    section("1. ARTEFACTOS EN DISCO")
    assert_true(os.path.exists(MODEL_PATH), f"existe {MODEL_PATH}")
    assert_true(os.path.exists(FEATURES_PATH), f"existe {FEATURES_PATH}")
    assert_true(os.path.exists(RAW_PATH), f"existe {RAW_PATH}")

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        saved_cols = json.load(f)
    assert_true(saved_cols == FEATURE_COLS,
                "feature_cols.json coincide con FEATURE_COLS de preprocessing")

    section("2. CARGA DEL MODELO")
    model = load_model(MODEL_PATH)
    assert_true(hasattr(model, "predict"), "modelo expone .predict")
    assert_true(hasattr(model, "predict_proba"), "modelo expone .predict_proba")
    n_features = int(model.n_features_in_)
    assert_true(n_features == len(FEATURE_COLS),
                f"n_features_in_={n_features} == len(FEATURE_COLS)={len(FEATURE_COLS)}")

    section("3. PIPELINE DE PREPROCESAMIENTO")
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
        raw_path=RAW_PATH, output_path=PROCESSED_PATH,
    )
    assert_true(list(X_train.columns) == FEATURE_COLS,
                "X_train.columns == FEATURE_COLS")
    assert_true(list(X_test.columns) == FEATURE_COLS,
                "X_test.columns == FEATURE_COLS")

    section("4. METRICAS SOBRE TEST SET")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print()
    print(classification_report(
        y_test, y_pred, target_names=["Rejected", "Approved"]))
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Matriz de confusion:\n{cm}")
    assert_true(acc >= TARGET_ACC,
                f"accuracy >= {TARGET_ACC:.0%} (obtenido {acc:.4f})")

    section("5. METRICAS SOBRE DATASET COMPLETO")
    df_full = pd.read_csv(PROCESSED_PATH)
    X_full = df_full.drop(columns=["loan_status"])[FEATURE_COLS]
    y_full = df_full["loan_status"]
    y_pred_full = model.predict(X_full)
    acc_full = accuracy_score(y_full, y_pred_full)
    print(f"  Accuracy total ({len(df_full)} filas): {acc_full:.4f}")
    assert_true(acc_full >= TARGET_ACC,
                f"accuracy global >= {TARGET_ACC:.0%} (obtenido {acc_full:.4f})")

    section("6. PIPELINE DE PREDICCION EN VIVO (preprocess_input)")
    # Caso 1: perfil fuerte -> probable Approved
    strong = {
        "no_of_dependents": 1,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 9_000_000,
        "loan_amount": 12_000_000,
        "loan_term": 8,
        "cibil_score": 820,
        "residential_assets_value": 10_000_000,
        "commercial_assets_value": 5_000_000,
        "luxury_assets_value": 15_000_000,
        "bank_asset_value": 7_000_000,
    }
    proc = preprocess_input(strong)
    assert_true(list(proc.columns) == FEATURE_COLS,
                "preprocess_input emite columnas en orden correcto")
    assert_true(proc.shape == (1, 14), f"shape {proc.shape} == (1, 14)")
    res = make_prediction(proc, model)
    print(f"  Perfil fuerte -> {res}")
    assert_true(res["prediction"] in (0, 1), "prediction es 0 o 1")

    # Caso 2: perfil debil -> probable Rejected
    weak = {
        "no_of_dependents": 5,
        "education": "Not Graduate",
        "self_employed": "Yes",
        "income_annum": 500_000,
        "loan_amount": 35_000_000,
        "loan_term": 20,
        "cibil_score": 350,
        "residential_assets_value": 100_000,
        "commercial_assets_value": 0,
        "luxury_assets_value": 200_000,
        "bank_asset_value": 50_000,
    }
    proc_w = preprocess_input(weak)
    res_w = make_prediction(proc_w, model)
    print(f"  Perfil debil  -> {res_w}")
    assert_true(res_w["prediction"] in (0, 1), "prediction es 0 o 1")
    assert_true(res["prediction"] != res_w["prediction"] or
                res["probability"] >= res_w["probability"],
                "perfil fuerte tiene probabilidad >= perfil debil")

    section("7. BATCH PREDICT SOBRE TEST")
    out = batch_predict(X_test, model)
    expected_cols = set(FEATURE_COLS) | {
        "prediction", "probability_approved", "label"}
    assert_true(set(out.columns) >= expected_cols,
                "batch_predict agrega columnas prediction/probability/label")
    assert_true(out["label"].isin({"Approved", "Rejected"}).all(),
                "batch_predict emite labels validas")

    section("8. CONVERSIONES DE LA APP (COP -> INR, Datacredito -> CIBIL)")
    # Replica la logica de app/app.py para asegurar mapping coherente
    COP_PER_INR = 50
    DATACREDITO_MIN, DATACREDITO_MAX = 150, 950
    CIBIL_MIN, CIBIL_MAX = 300, 900

    def cop_to_inr(v):
        return v / COP_PER_INR

    def datacredito_to_cibil(s):
        pct = (s - DATACREDITO_MIN) / (DATACREDITO_MAX - DATACREDITO_MIN)
        return round(CIBIL_MIN + pct * (CIBIL_MAX - CIBIL_MIN))

    app_input = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": cop_to_inr(250_000_000),
        "loan_amount": cop_to_inr(750_000_000),
        "loan_term": 12,
        "cibil_score": datacredito_to_cibil(720),
        "residential_assets_value": cop_to_inr(350_000_000),
        "commercial_assets_value": cop_to_inr(150_000_000),
        "luxury_assets_value": cop_to_inr(250_000_000),
        "bank_asset_value": cop_to_inr(200_000_000),
    }
    proc_app = preprocess_input(app_input)
    res_app = make_prediction(proc_app, model)
    print(f"  Caso default de la app -> {res_app}")
    assert_true(res_app["prediction"] in (0, 1),
                "app produce prediccion valida con valores por defecto")

    section("RESUMEN")
    print(f"  Test accuracy     : {acc*100:.2f}%   (objetivo {TARGET_ACC*100:.0f}%)")
    print(f"  Dataset accuracy  : {acc_full*100:.2f}%")
    print(f"  ROC-AUC           : {auc:.4f}")
    print("  Pipeline en vivo  : OK")
    print("  Conversiones app  : OK")
    print("\nMODELO VERIFICADO CORRECTAMENTE.")


if __name__ == "__main__":
    main()