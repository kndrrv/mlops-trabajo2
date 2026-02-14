import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Cargar datos
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# Preparar datos
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Preprocesamiento
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type',
                       'Residence_type', 'smoking_status',
                       'hypertension', 'heart_disease']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear pipeline con modelo
modelo = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        class_weight='balanced'
    ))
])

# Entrenar
print("Entrenando modelo...")
modelo.fit(X_train, y_train)
print("✓ Modelo entrenado")

# Validación cruzada
print("\nValidación cruzada (5-fold):")
cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='roc_auc')
print(f"ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Guardar modelo
joblib.dump(modelo, 'modelo_stroke.joblib')
print("\n✓ Modelo guardado: modelo_stroke.joblib")
