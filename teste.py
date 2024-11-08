import pandas as pd
from predict import predict

df_test = pd.read_parquet('df_test.parquet')
df_test = df_test.drop("Product", axis=1)
df_test = df_test.sample(5)

for i in range(min(5, len(df_test))): 
    data = df_test.iloc[[i]]
    print(f"Predição para amostra {i+1}:")
    print(predict(data), "ATR_DF: ", data['ATR'])
    print("\n")
