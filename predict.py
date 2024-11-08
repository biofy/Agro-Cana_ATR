import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

model = layers.TFSMLayer("model/modelo_final", call_endpoint='serving_default')

# Carregar o scaler e encoders
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("model/columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


def predict(data):
    data.iloc[:, 2:-5] = scaler.transform(data.iloc[:, 2:-5])

    # OneHotEncoder
    ohe_features = encoders['ohe'].transform(data[['Fazenda', 'Variedade', 'label']])
    ohe_df = pd.DataFrame(
        ohe_features, 
        columns=[f'{col}_{val}' for col, values in zip(['Fazenda', 'Variedade', 'label'], encoders['ohe'].categories_) for val in values]
    )

    data = pd.concat([data.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    data = data.drop(['Fazenda', 'Variedade', 'label', 'Product'], axis=1, errors='ignore')

    data = data.reindex(columns=model_columns, fill_value=0)

    data = np.array(data.astype(float))
    prediction = model(data)
    return prediction
