import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

# Cargar los datos
data_df = pd.read_csv("../../data/processed/TCGA_GBM_LGG_Mutations_clean_v2.csv")  # Cambiar la ruta al probar

# Preparar datos para entrenamiento
X = data_df.drop(["Grade"], axis=1)
y = data_df["Grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalamiento de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurar el experimento en MLflow
mlflow.set_experiment("Tumor_Classification")

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params, save_path):
    with mlflow.start_run(run_name=model_name):
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Realizar predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        # Registrar parámetros y métricas en MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})
        
        # Guardar el modelo en MLflow
        mlflow.sklearn.log_model(model, artifact_path="models")
        
        # Guardar el modelo en un archivo pickle
        with open(save_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Modelo guardado en {save_path}")

# Definir el modelo y los parámetros
params_lr = {"C": 1.0, "solver": "liblinear", "random_state": 42}
model_lr = LogisticRegression(**params_lr)

# Entrenar y registrar el modelo
train_and_log_model(
    model=model_lr,
    model_name="Logistic_Regression",
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    params=params_lr,
    save_path="logistic_regression_model.pkl"
)
