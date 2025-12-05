import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import optuna
import mlflow
import getpass

import sys
sys.stdout.reconfigure(encoding='utf-8')

REPO_NAME = "my-first-repo"
REPO_OWNER = "NoeCoatl19-SRR"
USER_NAME = "NoeCoatl19-SRR"

os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = "1cc8ef8f5688dac6c672772d03a6c2232165cda2"


mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')
mlflow.set_experiment("EfficientNetV2M_Optuna_T6")

# -------------------------------
# 1. CONFIGURACIÓN DE LOS DATOS
# -------------------------------
BASE_DIR = r"C:\Users\Noe\Desktop\plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(BASE_DIR, "images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")

IMG_SIZE = (224, 224)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# -------------------------------
# 2. CARGAMOS Y DIVIDIMOS LOS DATOS EN 30% TEST, 50% TRAIN, 20% VALIDACIÓN
# -------------------------------
df = pd.read_csv(TRAIN_CSV)
df['image_id'] = df['image_id'] + '.jpg'
df = df.drop_duplicates(subset='image_id').reset_index(drop=True)

labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
df['class'] = np.argmax(df[labels].values, axis=1)

df_train_val, df_test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['class'])
df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=SEED, stratify=df_train_val['class'])

# -------------------------------
# 3. DATASETS CON BATCH DE 32
# -------------------------------
def Creador_datasets(df_split, shuffle=False, batch_size=32):
    file_paths = [os.path.join(IMG_DIR, img_id) for img_id in df_split['image_id']]
    labels = df_split['class'].values
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def pre_procesamiento(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, tf.one_hot(label, 4)
    
    ds = ds.map(pre_procesamiento, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# 4. FUNCIÓN OBJETIVO PARA OPTUNA 
# -------------------------------
def objective(trial):
    with mlflow.start_run(nested=True):
        train_ds = Creador_datasets(df_train, shuffle=True)
        # --- Hiperparámetros ---
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        dropout1 = trial.suggest_float("dropout1", 0.3, 0.7)
        dropout2 = trial.suggest_float("dropout2", 0.1, 0.5)
        units = trial.suggest_categorical("units", [256, 512, 768, 1024])
        
        rot = trial.suggest_float("rotation", 0.1, 0.3)
        zoom = trial.suggest_float("zoom", 0.1, 0.3)
        contrast = trial.suggest_float("contrast", 0.1, 0.4)

        # --- Aumento de datos ---
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(rot),
            layers.RandomZoom(zoom),
            layers.RandomContrast(contrast),
            layers.RandomBrightness(0.2),
        ], name="aug")

        # --- Modelo EfficientNetV2M ---
        base_model = keras.applications.EfficientNetV2M(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False  # Fase sin fine-tuning

        inputs = keras.Input(shape=(*IMG_SIZE, 3))
        x = data_augmentation(inputs)
        x = keras.applications.efficientnet_v2.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout1)(x)
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout2)(x)
        outputs = layers.Dense(4, activation="softmax")(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # --- Callbacks ---
        filepath = f"best_effnet_trial_{trial.number}.keras"
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, monitor="loss", mode="min", verbose=1),
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="loss", mode="min")
        ]

        # --- Entrenamiento ---
        history = model.fit(
            train_ds,
            epochs=10,
            callbacks=callbacks,
            verbose=1
        )

        # === MÉTRICAS ===
        best_train_acc = max(history.history['accuracy'])
        best_train_loss = min(history.history['loss'])

        # === LOG EN MLFLOW ===
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_train_loss", best_train_loss)
        mlflow.log_metric("best_train_accuracy", best_train_acc)
        mlflow.log_artifact(filepath, artifact_path="models")

    return best_train_loss

# -------------------------------
# 5. EJECUTAMOS OPTUNA
# -------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  

print("MEJOR CONFIGURACIÓN:")
print(study.best_params)
