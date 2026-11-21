import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# --- Check device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load data ---
train_df = pd.read_csv("./date/train.csv")
test_df = pd.read_csv("./date/test.csv")

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=True)

# --- Frequency dictionary ---
token_freq = Counter(train_df['token'])
max_freq = max(token_freq.values())
token_freq = {k: v/max_freq for k, v in token_freq.items()}

# --- Dummy age_of_acquisition dictionary ---
age_of_acquisition_dict = {}  # pune date reale dacă ai

def get_age_of_acquisition(word):
    return age_of_acquisition_dict.get(word.lower(), 0.5)

def count_syllables(word):
    vowels = "aeiouăîâAEIOUĂÎÂ"
    return sum(ch in vowels for ch in word)

def count_consonants(word):
    vowels = "aeiouăîâAEIOUĂÎÂ"
    return sum(ch.isalpha() and ch not in vowels for ch in word)

def is_abbreviation(word):
    return 1.0 if '.' in word else 0.0

def word_length(word):
    return len(word)

def extract_features(token):
    features = [
        word_length(token),
        count_syllables(token),
        count_consonants(token),
        token_freq.get(token, 0.0),
        get_age_of_acquisition(token),
        is_abbreviation(token)
    ]
    return torch.tensor(features, dtype=torch.float32)

# --- Dataset ---
class ComplexityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128, is_test=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.samples = []

        for _, row in dataframe.iterrows():
            sentence = row['sentence']
            token = row['token']
            marked = sentence.replace(token, f"[TGT] {token} [TGT]", 1)

            encoded = tokenizer(
                marked,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            features = extract_features(token)

            item = {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'features': features
            }

            if not is_test:
                item['labels'] = torch.tensor(row['score'], dtype=torch.float)

            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- Collate function for dataloader ---
def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    features = torch.stack([x['features'] for x in batch])
    if 'labels' in batch[0]:
        labels = torch.stack([x['labels'] for x in batch])
        return input_ids, attention_mask, features, labels
    else:
        return input_ids, attention_mask, features

# --- Model ---
class BertWithFeatures(nn.Module):
    def __init__(self, model_name, feature_dim=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size + feature_dim, 1)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_emb, features], dim=1)
        combined = self.dropout(combined)
        out = self.linear(combined)
        return out.squeeze(-1)

# --- Init tokenizer and model ---
model_name = "dumitrescustefan/bert-base-romanian-cased-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(["[TGT]"])

model = BertWithFeatures(model_name=model_name).to(device)
model.bert.resize_token_embeddings(len(tokenizer))

# --- Datasets and loaders ---
train_dataset = ComplexityDataset(train_df, tokenizer)
val_dataset = ComplexityDataset(val_df, tokenizer)
test_dataset = ComplexityDataset(test_df, tokenizer, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# --- Optimizer and loss ---
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.MSELoss()

# --- Training loop ---
def train_epoch():
    model.train()
    losses = []
    for input_ids, attention_mask, features, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# --- Evaluation loop ---
def evaluate(loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for input_ids, attention_mask, features, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, features)
            preds.extend(outputs.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    r2 = r2_score(trues, preds)
    spearman_corr = spearmanr(trues, preds).correlation
    return r2, spearman_corr

# --- Main training ---
EPOCHS = 4
for epoch in range(EPOCHS):
    train_loss = train_epoch()
    r2, spearman = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val R2: {r2:.4f} - Val Spearman: {spearman:.4f}")

# --- Predict test ---
model.eval()
all_preds = []
with torch.no_grad():
    for input_ids, attention_mask, features in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        features = features.to(device)

        outputs = model(input_ids, attention_mask, features)
        all_preds.extend(outputs.cpu().numpy())

all_preds = np.clip(all_preds, 0, 1)

# --- Save submission ---
output_df = pd.DataFrame({
    "Id": test_df["Id"],
    "complexity": all_preds
})
output_df.to_csv("submission.csv", index=False)
print("✅ 'submission.csv' has been generated.")
