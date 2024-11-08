import os
import random
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder #, LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle

seed_value = 23
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.config.experimental.enable_op_determinism()
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

df_ = pd.read_parquet('df_cana_ATR.parquet')
df = df_.drop('Product', axis=1).copy()

scaler = MinMaxScaler()
df.iloc[:, 2:-5] = scaler.fit_transform(df.iloc[:, 2:-5])

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_features = ohe.fit_transform(df[['Fazenda', 'Variedade', 'label']])
ohe_colunas = [f'{col}_{valor}' for col, valores in zip(['Fazenda', 'Variedade', 'label'], ohe.categories_) for valor in valores]
df_ohe = pd.DataFrame(ohe_features, columns=ohe_colunas)
df = pd.concat([df, df_ohe], axis=1)
df = df.drop(['Fazenda', 'Variedade', 'label'], axis=1)

target_col = 'ATR'
drop_x_cols = ['ID', 'Sample', target_col]
X = df.drop(drop_x_cols, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

with open("model/columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

n_neurons_1 = 576
n_neurons_2 = 3216
n_neurons_3 = 1938
n_neurons_4 = 814
learning_rate = 0.00043665389199997286
batch_size = 175
epochs = 150

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(n_neurons_1, activation='relu'),
    keras.layers.Dense(n_neurons_2, activation='relu'),
    keras.layers.Dense(n_neurons_3, activation='relu'),
    keras.layers.Dense(n_neurons_4, activation='relu'),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mape', metrics=['mae', 'mse'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

loss = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Loss no conjunto de teste: {loss}")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"R-squared: {r2}")
print(f"MSE: {mse}")
print(f"MAPE: {mape}")

try:
    experiment_id = mlflow.create_experiment("my_experimento")
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name("my_experimento")
    experiment_id = experiment.experiment_id

with mlflow.start_run(experiment_id=experiment_id, run_name="my_run"):
    params = {
        "n_neurons_1": n_neurons_1,
        "n_neurons_2": n_neurons_2,
        "n_neurons_3": n_neurons_3,
        "n_neurons_4": n_neurons_4,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    mlflow.log_params(params)

    metrics = {
        "r2": r2,
        "mse": mse,
        "mape": mape
    }
    mlflow.log_metrics(metrics)

    artifact_path = "model"
    os.makedirs(artifact_path, exist_ok=True)

    model_dir = os.path.join(artifact_path, "modelo_final") 
    tf.saved_model.save(model, model_dir)
    mlflow.log_artifacts(model_dir, artifact_path=artifact_path)

    scaler_path = os.path.join(artifact_path, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact(scaler_path, artifact_path=artifact_path)

    encoder_path = os.path.join(artifact_path, "encoders.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump({ 'ohe': ohe}, f)
    mlflow.log_artifact(encoder_path, artifact_path=artifact_path)

    mlflow.log_artifact("predict.py", artifact_path=artifact_path)

print(mlflow.get_artifact_uri(artifact_path))
