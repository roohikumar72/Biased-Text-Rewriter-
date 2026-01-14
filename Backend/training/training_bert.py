# import pandas as pd
# import torch
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# df = pd.read_csv("../data/demo_bias_data.csv")
# label_map = {"biased": 0, "inclusive": 1}
# df["label"] = df["label"].map(label_map)

# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     df["text"], df["label"], test_size=0.2
# )

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# def tokenize(texts):
#     return tokenizer(list(texts), truncation=True, padding=True, max_length=128)

# train_enc = tokenize(train_texts)
# val_enc = tokenize(val_texts)

# class BiasDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.enc = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
#         item["labels"] = torch.tensor(self.labels.iloc[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_ds = BiasDataset(train_enc, train_labels)
# val_ds = BiasDataset(val_enc, val_labels)

# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=2
# )

# args = TrainingArguments(
#     output_dir="../models/bert-bias",
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     save_strategy="epoch"
# )

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds
# )

# trainer.train()

# model.save_pretrained("../models/bert-bias")
# tokenizer.save_pretrained("../models/bert-bias")

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
from preprocess import load_data

MODEL_NAME = "bert-base-uncased"

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Load data
train_texts, val_texts, train_labels, val_labels = load_data(
    "../data/sample_data.csv"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Create HF datasets
train_dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels
}).map(tokenize, batched=True)

val_dataset = Dataset.from_dict({
    "text": val_texts,
    "label": val_labels
}).map(tokenize, batched=True)

# Load model
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ✅ FIXED ARGUMENT NAME
training_args = TrainingArguments(
    output_dir="./bert-bias",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
trainer.save_model("./bert-bias")
tokenizer.save_pretrained("./bert-bias")
