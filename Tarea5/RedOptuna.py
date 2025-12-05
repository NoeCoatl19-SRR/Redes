import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
import mlflow
import os
from getpass import getpass
from keras.callbacks import ModelCheckpoint, EarlyStopping

import sys
sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# CONEXIÓN A DAGSHUB
# =============================================================================
REPO_NAME = "my-first-repo"
REPO_OWNER = "NoeCoatl19-SRR"
USER_NAME = "NoeCoatl19-SRR"

os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = "1cc8ef8f5688dac6c672772d03a6c2232165cda2"

mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

mlflow.tensorflow.autolog()

# =============================================================================
# CARGA DE DATOS
# =============================================================================
df = pd.read_csv(r"C:\Users\Noe\Downloads\global_house_purchase_dataset.csv")
df = df.drop("property_id", axis=1)

categorical_cols = ["country", "city", "property_type", "furnishing_status"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)

X = df_encoded.drop("decision", axis=1).values
y = df_encoded["decision"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test / Validation (70% train, 20% test usado como val, 10% val usado como test final)
X_train, X_pv, y_train, y_pv = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_pv, y_pv, test_size=1/3, random_state=42)

num_features = X_train.shape[1]
print("El número de características es:", num_features)

# Calcular class weights para manejar desbalanceo
classes = np.unique(y_train)
class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# =============================================================================
# MODELO DENSO
# =============================================================================
def create_model(trial):

    inputs = keras.Input(shape=(num_features,))
    x = inputs

    n_layers = trial.suggest_int("n_layers", 1, 5)

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
    model = Model(inputs, outputs)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    opt = {
        "adam": keras.optimizers.Adam(lr),
        "sgd": keras.optimizers.SGD(lr),
        "rmsprop": keras.optimizers.RMSprop(lr),
    }[optimizer_name]

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    return model

# =============================================================================
# OBJETIVO OPTUNA
# =============================================================================
def objective(trial):

    with mlflow.start_run(nested=True):

        model = create_model(trial)

        batch_size = 128
        epochs = 15  # Aumentado para dar más margen

        checkpoint = ModelCheckpoint(
            "best_model_dense.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        )

        earlystop = EarlyStopping(
            monitor="val_loss",
            patience=10,  # Aumentado para evitar paradas prematuras
            min_delta=0.001,  # Ignora cambios muy pequeños
            restore_best_weights=True
        )

        # ENTRENAR usando TEST como validación, con class weights
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,  
            callbacks=[checkpoint, earlystop],
            class_weight=class_weight_dict  # Añadido para penalizar clase desbalanceada
        )

        # Log del número de épocas entrenadas
        num_epochs = len(history.history['loss'])
        mlflow.log_metric("num_epochs", num_epochs)

        # EVALUACIÓN en validación (X_test)
        min_val_loss = min(history.history['val_loss'])
        max_val_acc = max(history.history['val_accuracy'])  # Corregido: 'val_accuracy'
        mlflow.log_metric("min_val_loss", min_val_loss)  # Corregido: usa el valor real
        mlflow.log_metric("max_val_acc", max_val_acc)  # Corregido: usa el valor real

        # Optuna optimiza la acc (maximizar)
        return max_val_acc

# =============================================================================
# OPTUNA STUDY
# =============================================================================
mlflow.set_experiment("dense_optuna")

study = optuna.create_study(direction="maximize")  # Corregido: para maximizar acc
study.optimize(objective, n_trials=20)

print("Trials terminados:", len(study.trials))
best_trial = study.best_trial
print("Mejor trial:", best_trial)
print("Loss:", best_trial.value)
print("Hiperparámetros:", best_trial.params)

# =============================================================================
# ENTRENAMIENTO FINAL CON MEJORES HIPERPARÁMETROS Y EVALUACIÓN EN TEST (X_val)
# =============================================================================
print("\nEntrenando modelo final con mejores hiperparámetros...")

# Crea un trial fijo con los mejores params
fixed_trial = optuna.trial.FixedTrial(best_trial.params)
best_model = create_model(fixed_trial)

# Callbacks para el entrenamiento final (consistente con acc: monitor val_accuracy)
earlystop_final = EarlyStopping(
    monitor="val_accuracy",  # Opcional: si priorizas acc; sino usa "val_loss" con mode="min"
    patience=10,
    min_delta=0.001,
    mode="max",
    restore_best_weights=True
)

# Entrena en train, valida en "test" (usado como val), con class weights
history_final = best_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=128,
    epochs=50,
    verbose=1,
    callbacks=[earlystop_final],
    class_weight=class_weight_dict  # Añadido para penalizar clase desbalanceada
)

# Evaluación final en "validación" usada como test (X_val)
test_loss, test_acc = best_model.evaluate(X_val, y_val, verbose=1)
print(f"\nPérdida en test (X_val): {test_loss}")
print(f"Precisión en test (X_val): {test_acc}")