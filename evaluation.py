import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib

RANDOM_STATE = 42

# Cargar datos
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Cargar modelo
modelo = joblib.load('modelo_stroke.joblib')
print("✓ Modelo cargado")

# Predecir
y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas
print("\n=== RESULTADOS DEL MODELO ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# Matriz de confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred,
                          target_names=['No Stroke', 'Stroke']))

# Predicción de prueba
print("\nPredicción de prueba:")
prediccion = modelo.predict(X_test.head(1))
print(f"Resultado: {'Stroke' if prediccion[0] == 1 else 'No Stroke'}")