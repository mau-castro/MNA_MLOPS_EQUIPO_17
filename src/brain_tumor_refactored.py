import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')
filepath = "../data/raw/TCGA_GBM_LGG_Mutations_all.csv"

# Loading the data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Función para convertir 'years days' a número decimal
def convertir_edad_decimal(age_str):
    try:
        if not "days" in age_str:
            return float(age_str.split(' years')[0])
        else:
            if ' years ' in age_str:
                years, days = age_str.split(' years ')
                years = int(years)
                days = int(days.split(' days')[0])
                return round(years + days / 365.25, 2)  # Redondear a 2 decimales
    except ValueError:
        # En caso de que no se tenga el valor de la edad, devolver None
        return None

# Función para extraer el tipo de tumor y la especificación del tumor
def extraer_tumor_info(diagnosis):
    if diagnosis == '--' or diagnosis not in ['Oligodendroglioma, NOS', 'Glioblastoma' , 'Mixed glioma', 'Astrocytoma, NOS', 'Astrocytoma, anaplastic', 'Oligodendroglioma, anaplastic']:
        return pd.Series([None, None])
    else:
        parts = diagnosis.split(', ')
        tumor_type = parts[0]
        tumor_specification = parts[1] if len(parts) > 1 else None
        return pd.Series([tumor_type, tumor_specification])

def variables_a_binario(dataset):
    # Eliminamos las filas con valores nulos en las column
    dataset.dropna(inplace=True)
    # Convertir variables binarias a 0 y 1
    dataset.replace({'NOT_MUTATED': 0, 'MUTATED': 1}, inplace=True)
    dataset.replace({'Female': 0, 'Male': 1}, inplace=True)
    dataset.replace({'LGG': 0, 'GBM': 1}, inplace=True)
    return dataset

def creacion_tumor_info(dataset):
    # Aplicar la función a la columna 'Primary_Diagnosis'
    dataset[['Tumor_Type', 'Tumor_Specification']] = dataset['Primary_Diagnosis'].apply(extraer_tumor_info)
    # Eliminar filas con 'Primary_Diagnosis' como None o '--'
    dataset = dataset.dropna(subset=['Tumor_Type'])
    return dataset

def mapeo_variables(dataset):
    race_mapping = {
    'not reported': 0,
    '--': 0,
    'white': 1,
    'black or african american': 2,
    'asian': 3,
    'american indian or alaska native': 4
    }
    mapeo_tumor = {
    "Oligodendroglioma": 0,
    "Mixed glioma": 1,
    "Astrocytoma": 2,
    "Glioblastoma": 3
    }
    mapeo_tipo_tumor = {
        None: 0,
        "NOS": 1,
        "anaplastic": 2
    }
    # Aplicar el mapeo a la columna Race
    dataset['Race'] = dataset['Race'].map(race_mapping)
    # Eliminar filas con 'Primary_Diagnosis' como None o '--'
    dataset = dataset.dropna(subset=['Tumor_Type'])
    # Aplicar el mapeo a las columnas 'Tumor_Type' y 'Tumor_Specification'
    dataset['Tumor_Type'] = dataset['Tumor_Type'].map(mapeo_tumor)
    dataset['Tumor_Specification'] = dataset['Tumor_Specification'].map(mapeo_tipo_tumor)
    # Convertir la columna 'Age_at_diagnosis' a número decimal
    dataset['Age_at_diagnosis'] = dataset['Age_at_diagnosis'].apply(convertir_edad_decimal)
    return dataset

def eliminar_variables_innecesarias(dataset):
    # Eliminar la columna 'Primary_Diagnosis' ya que no se utilizará
    dataset.drop(columns=['Primary_Diagnosis'], inplace=True)
    # ya que el proyecto se puede sacar con el grade y el case Id son indiferentes para el analisis de datos posterior
    dataset.drop(columns=['Project', 'Case_ID'], inplace=True)
    return dataset

def ordenar_columnas(dataset):
    column_order = ['Grade', 'Gender', 'Age_at_diagnosis', 'Race', 'Tumor_Type', 'Tumor_Specification' , 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
    dataset = dataset[column_order]
    return dataset

def guardar_csv(dataset, save_path):
    # Salvar el dataframe limpio en un nuevo archivo csv
    dataset.to_csv(save_path, index=False)

def clean_csv(dataset, save_path):
    dataset = variables_a_binario(dataset)
    dataset = creacion_tumor_info(dataset)
    dataset = mapeo_variables(dataset)
    dataset = eliminar_variables_innecesarias(dataset)
    dataset = ordenar_columnas(dataset)
    guardar_csv(dataset, save_path)
    return dataset

# Splitting el dataset
def split_data(data, target, test_size=0.2, random_state=42):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

#Entrenando el modelo
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=44)
    model.fit(X_train, y_train)
    return model

# Evaluando el modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

# Validación cruzada
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print("Average accuracy with CV:", np.mean(scores))

# Main function to run the pipeline
def main(filepath):
    data = load_data(filepath)
    dataset = clean_csv(data, "../data/raw/TCGA_GBM_LGG_Mutations_clean.csv")
    X_train, X_test, y_train, y_test = split_data(dataset, 'Grade')
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    cross_validate_model(model, dataset.drop('Grade', axis=1), dataset['Grade'])

main(filepath)