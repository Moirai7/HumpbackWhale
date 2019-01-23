import torch
import pandas as pd
import collections


df = pd.read_csv("../dataset/label.csv")

df = df.sample(frac=1)
print(df)

