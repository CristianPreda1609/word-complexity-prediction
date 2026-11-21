
from gensim.models import KeyedVectors
import numpy as np
from sklearn.model_selection import train_test_split

# Load FastText embeddings
embedding_path = "./vec/wiki.ro.vec"  # Download from https://fasttext.cc/docs/en/crawl-vectors.html
embeddings = KeyedVectors.load_word2vec_format(embedding_path)

embedding_dim = 300

import pandas as pd
import torch
from torch.utils.data import Dataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load train and test data
train_df = pd.read_csv("./date/train.csv")
test_df = pd.read_csv("./date/test.csv")

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=True)

class EmbeddingDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.sentences = df["sentence"].values
        self.tokens = df["token"].values
        self.labels = df["score"].values if not is_test else None
        self.is_test = is_test

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        token = self.tokens[idx].lower()
        vec = embeddings[token] if token in embeddings else np.zeros(embedding_dim)

        if self.is_test:
            return torch.tensor(vec, dtype=torch.float32)
        else:
            label = self.labels[idx]
            return torch.tensor(vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

import torch.nn as nn

class ComplexityRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ComplexityRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

from torch.utils.data import DataLoader
import torch.optim as optim

model = ComplexityRegressor(embedding_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_dataset = EmbeddingDataset(train_df)
val_dataset = EmbeddingDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

test_dataset = EmbeddingDataset(test_df, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=64)

model.eval()
preds = []
with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        output = model(x)
        preds.extend(output.cpu().numpy())

# Save submission
submission = pd.DataFrame({"Id": test_df["Id"], "complexity": np.clip(preds, 0, 1)})
submission.to_csv("submission.csv", index=False)
