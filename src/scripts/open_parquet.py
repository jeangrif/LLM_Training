import pandas as pd
file_path = "/Users/jeangrifnee/PycharmProjects/LLMTraining/data/raw/squad_train.parquet"

df = pd.read_parquet(file_path)
print(df.columns)
print(df["answers"].head(5))
print(df["answers"].iloc[0]["answer_start"])