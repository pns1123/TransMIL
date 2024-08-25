import pandas as pd

labels = [
    {"filename": f"test{k}", "label0": 0, "label1": 1, "group": group}
    for k in range(10)
    for group in ["train", "test", "validation"]
]

print(labels)
df = pd.DataFrame.from_records(labels)
df.to_csv("test_labels.csv")
