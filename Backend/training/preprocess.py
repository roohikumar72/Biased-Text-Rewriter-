import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

# def load_data(path):
#     df = pd.read_csv(path)
#     return train_test_split(df["text"], df["label"], test_size=0.2)

def load_data(path):
    df = pd.read_csv(path)

    df["label"] = df["label"].astype(int)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    return train_texts, val_texts, train_labels, val_labels