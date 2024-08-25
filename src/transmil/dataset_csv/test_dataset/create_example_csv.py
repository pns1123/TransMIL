import pandas as pd

labels = [
    {"filename": "test0", "label0": 0, "label1": 1, "group": "train"},
    {"filename": "test1", "label0": 0, "label1": 0, "group": "validation"},
    {"filename": "test2", "label0": 1, "label1": 1, "group": "test"},
]

df = pd.DataFrame.from_records(labels)
print(df.head())
df.to_csv("test_labels.csv")
