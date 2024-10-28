from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import sys 
import joblib

def read_data(path):
    data_df = pd.read_csv(path)
    return data_df



def train_model_v1(data_df):
    # Separar nuestros datos
    X = data_df.drop(["Grade"], axis=1)
    y = data_df["Grade"]
    # Generar los datos para probar y para entrenar con parametros seleccionados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
    # Generar un modelo Random Forest con hiperparámetros seleccionados
    rf_model = RandomForestClassifier(n_estimators=50, random_state=44)
    # Entrenar el módelo con nuestros sets de entrenamiento
    rf_model.fit(X_train, y_train)
    # Ahora podemos obtener los valores de importancia de acuerdo con el modelo
    importances = rf_model.feature_importances_
    columns = X.columns
    i = 0
    while i < len(columns):
        print(f" The importance of feature '{columns[i]}' is {round(importances[i] * 100, 2)}%.")
        i += 1
    return rf_model, X_train, X_test, y_train, y_test

def print_predictions(rf_model, X_test):
    print(rf_model.predict(X_test))
    print(rf_model.predict_proba(X_test))

def grid_classif(rf_model, X_train, X_test, y_train, y_test):
    y_true, y_pred = y_test, rf_model.predict(X_test)
    print(classification_report(y_true, y_pred))
    # Generamos un Random Forest para poder encontrar los hiperparámetros adecuados
    clf = RandomForestClassifier(n_estimators=50, random_state=44)
    param_grid = {
        'n_estimators': [5, 10, 15, 20],
        'max_depth': [2, 5, 7, 9]
    }
    grid_clf = GridSearchCV(clf, param_grid, cv=10)
    grid_clf.fit(X_train, y_train)
    print(grid_clf.cv_results_)

def train():
    path = 'path_to_your_data.csv'  # Reemplaza con la ruta a tu archivo de datos
    data_df = read_data(path)
    rf_model, X_train, X_test, y_train, y_test = train_model(data_df)
    print_predictions(rf_model, X_test)
    grid_classif(rf_model, X_train, X_test, y_train, y_test)
    y_true, y_pred = y_test, rf_model.predict(X_test)
    return y_true, y_pred


def train_model(X_train,y_train):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    # Generar un modelo Random Forest con hiperparámetros seleccionados
    rf_model = RandomForestClassifier(n_estimators=50, random_state=44)
    # Entrenar el módelo con nuestros sets de entrenamiento
    rf_model.fit(X_train, y_train)
    # Ahora podemos obtener los valores de importancia de acuerdo con el modelo
    return rf_model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    model = train_model(X_train_path, y_train_path)
    joblib.dump(model, model_path)