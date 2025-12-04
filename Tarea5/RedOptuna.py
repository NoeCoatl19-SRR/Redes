import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
import mlflow
import os
from getpass import getpass
from keras.callbacks import ModelCheckpoint, EarlyStopping

import sys
sys.stdout.reconfigure(encoding='utf-8')

REPO_NAME= "my-first-repo"
REPO_OWNER= "NoeCoatl19-SRR"  
USER_NAME = "NoeCoatl19-SRR" 

os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = "1cc8ef8f5688dac6c672772d03a6c2232165cda2"
mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')


# =============================================================================
# CARGA DE DATOS
# =============================================================================
df = pd.read_csv(r"C:\Users\Noe\Downloads\global_house_purchase_dataset.csv")
df = df.drop('property_id', axis=1)

categorical_cols = ['country', 'city', 'property_type', 'furnishing_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)

# Features y Labels
X = df_encoded.drop('decision', axis=1).values
y = df_encoded['decision'].values

# Normalización
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test / Validation
X_train, X_pv, y_train, y_pv = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_pv, y_pv, test_size=1/3, random_state=42)

num_features = X_train.shape[1]
print("El número de características es: ", num_features)

# =============================================================================
# MODELO DENSO PARA OPTUNA
# =============================================================================

def create_model(trial):

    inputs = keras.Input(shape=(num_features,))

    # Número de capas ocultas
    n_layers = trial.suggest_int("n_layers", 1, 5)

    x = inputs

    for i in range(n_layers):
        units = trial.suggest_int(f"units_layer_{i}", 64, 512, step=64)
        dropout_rate = trial.suggest_float(f"dropout_{i}", 0.0, 0.5)

        l1_value = trial.suggest_float(f"l1_{i}", 1e-6, 1e-3, log=True)

        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l1(l1_value)
        )(x)

        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Optimizador
    optimizer_selected = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    if optimizer_selected == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_selected == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def objective(trial):

    with mlflow.start_run(nested=True):

        model = create_model(trial)

        batch_size = trial.suggest_int("batch_size", 32, 256, step=32)

        epochs = 20  # puedes subirlo si quieres

        # Callbacks
        filepath = "best_model_dense.keras"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", save_best_only=True, mode="min")
        earlystop = EarlyStopping(monitor="loss", mode="min", patience=5, restore_best_weights=True)

        # ENTRENAMIENTO
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[checkpoint, earlystop]
        )

        # EVALUACIÓN EN TEST
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Logging en MLflow
        mlflow.log_params(trial.params)
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(filepath, artifact_path="models")

    return loss

mlflow.set_experiment("dense_optuna")

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Número de pruebas terminadas: ", len(study.trials))
best_trial = study.best_trial
print("Mejor intento: ", best_trial)
print("Valor (loss): ", best_trial.value)
print("Hiperparámetros: ", best_trial.params)
