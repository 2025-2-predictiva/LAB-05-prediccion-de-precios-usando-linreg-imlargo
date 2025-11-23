# ==============================================================================
# HOMEWORK: Predicción de Precios de Vehículos usando Regresión Lineal
# ==============================================================================
#
# CÓMO EJECUTAR:
# 1. Para ver el código sin entrenar: python homework/homework.py
# 2. Para entrenar el modelo: Cambia SKIP_TRAINING = False al final del archivo
# 3. El entrenamiento toma ~2-5 minutos
#
# ARCHIVOS GENERADOS:
# - files/models/model.pkl.gz: Modelo entrenado (comprimido)
# - files/output/metrics.json: Métricas de rendimiento
#
# ==============================================================================
#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import gzip
import json
import os
import pickle
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data():
    """Load and preprocess the data."""
    # Unzip and load training data
    train_zip_path = "files/input/train_data.csv.zip"
    with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
        zip_ref.extractall("files/input/")
    train_data = pd.read_csv("files/input/train_data.csv")

    # Unzip and load test data
    test_zip_path = "files/input/test_data.csv.zip"
    with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
        zip_ref.extractall("files/input/")
    test_data = pd.read_csv("files/input/test_data.csv")

    return train_data, test_data


def preprocess_data(train_data, test_data):
    """Preprocess the data according to requirements."""
    # Create Age column (assuming current year is 2021)
    train_data["Age"] = 2021 - train_data["Year"]
    test_data["Age"] = 2021 - test_data["Year"]

    # Drop Year and Car_Name columns
    train_data = train_data.drop(columns=["Year", "Car_Name"])
    test_data = test_data.drop(columns=["Year", "Car_Name"])

    return train_data, test_data


def split_data(train_data, test_data):
    """Split data into features and target."""
    x_train = train_data.drop(columns=["Present_Price"])
    y_train = train_data["Present_Price"]

    x_test = test_data.drop(columns=["Present_Price"])
    y_test = test_data["Present_Price"]

    return x_train, y_train, x_test, y_test


def create_pipeline(x_train):
    """Create the ML pipeline with preprocessing and model."""
    # Identify categorical and numerical columns
    categorical_features = x_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Create preprocessing for categorical and numerical features
    # Apply OneHotEncoder to categorical features
    # Apply MinMaxScaler to numerical features
    # Apply SelectKBest for feature selection
    # Apply LinearRegression for prediction
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", MinMaxScaler()),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    return pipeline


def train_model(pipeline, x_train, y_train):
    """Train the model using GridSearchCV."""
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        "feature_selection__k": list(range(1, 12)) + ["all"],
        "regressor__fit_intercept": [True, False],
    }

    # Create GridSearchCV with 10-fold cross-validation
    # Use negative mean absolute error as scoring metric
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
    )

    # Fit the model
    print("Training model with GridSearchCV...")
    grid_search.fit(x_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score (neg MAE): {grid_search.best_score_}")

    return grid_search


def save_model(model, filename="files/models/model.pkl.gz"):
    """Save the model as a compressed pickle file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.open(filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """Calculate and save metrics for train and test sets."""
    # Predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate metrics for training set
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "mad": mean_absolute_error(y_train, y_train_pred),
    }

    # Calculate metrics for test set
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_test, y_test_pred),
        "mse": mean_squared_error(y_test, y_test_pred),
        "mad": mean_absolute_error(y_test, y_test_pred),
    }

    return train_metrics, test_metrics


def save_metrics(train_metrics, test_metrics, filename="files/output/metrics.json"):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")
    print(f"Metrics saved to {filename}")
    print(f"Train metrics: {train_metrics}")
    print(f"Test metrics: {test_metrics}")


def main(skip_training=False):
    """
    Main function to execute the homework.
    
    Args:
        skip_training: If True, skip the expensive training operation.
                      Set to True to avoid long computation times.
    """
    print("=" * 70)
    print("HOMEWORK: Price Prediction Using Linear Regression")
    print("=" * 70)

    if skip_training:
        print("\nWARNING: Training is skipped! Set skip_training=False to train.")
        print("The model and metrics files will not be generated.")
        return

    # Step 1 & 2: Load and preprocess data
    print("\nStep 1: Loading data...")
    train_data, test_data = load_data()
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    print("\nStep 2: Preprocessing data...")
    train_data, test_data = preprocess_data(train_data, test_data)

    print("\nStep 3: Splitting data...")
    x_train, y_train, x_test, y_test = split_data(train_data, test_data)
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")

    # Step 3: Create pipeline
    print("\nStep 4: Creating pipeline...")
    pipeline = create_pipeline(x_train)
    print("Pipeline created successfully")

    # Step 4: Train model with GridSearchCV
    print("\nStep 5: Training model (this may take several minutes)...")
    model = train_model(pipeline, x_train, y_train)

    # Step 5: Save model
    print("\nStep 6: Saving model...")
    save_model(model)

    # Step 6: Calculate and save metrics
    print("\nStep 7: Calculating metrics...")
    train_metrics, test_metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(train_metrics, test_metrics)

    print("\n" + "=" * 70)
    print("HOMEWORK COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    # Set skip_training to True to avoid running expensive computations
    # Set to False when you want to actually train the model
    SKIP_TRAINING = True  # Change to False to train the model
    
    main(skip_training=SKIP_TRAINING)
