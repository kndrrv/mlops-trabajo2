import pandas as pd

# Cargar los datos
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print("\nPrimeras filas:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())