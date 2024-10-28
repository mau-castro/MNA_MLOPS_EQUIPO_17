import pandas as pd
import sys
from sklearn.model_selection import train_test_split

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

def clean_csv(dataset):
    dataset = variables_a_binario(dataset)
    dataset = creacion_tumor_info(dataset)
    dataset = mapeo_variables(dataset)
    dataset = eliminar_variables_innecesarias(dataset)
    dataset = ordenar_columnas(dataset)
    #guardar_csv(dataset, save_path)
    return dataset


def preprocess_data(data_path):
    dataset = pd.read_csv(data_path)
    data = clean_csv(dataset)
    # Separar nuestros datos
    X = data.drop(["Grade"], axis=1)
    y = data["Grade"]
    # Generar los datos para probar y para entrenar con parametros seleccionados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)


    return X_train, X_test, y_train, y_test


   



if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)