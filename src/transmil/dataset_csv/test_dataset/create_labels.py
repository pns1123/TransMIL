import pandas as pd
import numpy as np

labels = [
    {
        "filename": f"test{k}",
        "label0": np.random.randint(2),
        "label1": np.random.randint(2),
        "group": group,
    }
    for k in range(1, 11)
    for group in ["train", "test", "validation"]
]

print(labels)
df = pd.DataFrame.from_records(labels)
df.to_csv("test_labels.csv")
