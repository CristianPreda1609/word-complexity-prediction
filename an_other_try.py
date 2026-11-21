import os
import pandas as pd
import numpy as np
from collections import Counter # Deși nu mai e folosit direct, păstrăm pentru consistență
from tqdm import tqdm # Pentru bare de progres
from sklearn.metrics import r2_score # Pentru evaluare
from scipy.stats import pearsonr # Pentru evaluare
# Nu mai e necesar StandardScaler, deoarece BERT procesează direct textul
import torch # Biblioteca PyTorch
import torch.nn as nn # Module neuronale din PyTorch
from torch.utils.data import Dataset, DataLoader # Pentru a gestiona datele pentru PyTorch
from transformers import AutoTokenizer, AutoModel # Pentru a lucra cu modele Transformer
import nltk

# Asigură-te că nltk.punkt este descărcat
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. Setări inițiale și încărcare date ---
BASE_DIR = './date'

try:
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
except FileNotFoundError as e:
    print(f"Eroare la citirea fișierelor: {e}. Asigură-te că train.csv și test.csv sunt în directorul '{BASE_DIR}'.")
    exit()

# Verifică disponibilitatea GPU-ului
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Folosim dispozitivul: {DEVICE}")

# Numele modelului BERT românesc
BERT_MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"

# Inițializăm tokenizer-ul BERT
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

# --- 2. Funcția de evaluare ---
def pearson_r2(preds, y_true):
    """
    Calculează media dintre valoarea absolută a coeficientului de corelație Pearson
    și scorul R2, trunchiat la 0 dacă este negativ.
    """
    r2 = max(0, r2_score(y_true, preds))
    pears = pearsonr(y_true, preds)[0]
    pears = np.nan_to_num(pears) # Tratează cazurile unde Pearson ar putea fi NaN
    return (abs(pears) + r2) / 2

# --- 3. Clasa Dataset personalizată pentru PyTorch ---
class TextComplexityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128, has_labels=True):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentence = str(row['sentence'])
        
        # Tokenizează propoziția
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Adăugăm 'labels' (scorul) DOAR dacă dataset-ul are etichete (e setul de antrenament)
        if self.has_labels:
            score = torch.tensor(row['score'], dtype=torch.float)
            item['labels'] = score

        return item

# --- 4. Definirea modelului BERT pentru Regresie ---
class BertForRegression(nn.Module):
    def __init__(self, bert_model_name):
        super(BertForRegression, self).__init__()
        # Încărcăm modelul BERT pre-antrenat
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # Adăugăm un strat liniar la sfârșit pentru a prezice scorul
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Procesează input-urile prin modelul BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Obținem reprezentarea agregată a întregii propoziții (de obicei de la token-ul [CLS])
        pooled_output = outputs.pooler_output
        # Aplicăm stratul de regresie
        score = self.regressor(pooled_output)
        # Eliminăm dimensiunea extra de 1 (ex: [batch_size, 1] devine [batch_size])
        return score.squeeze(-1)

# --- 5. Parametri și inițializarea DataLoader-elor și Modelului ---
EPOCHS = 3 # Câte epoci de antrenament
BATCH_SIZE = 16 # Dimensiunea loturilor
LEARNING_RATE = 2e-5 # Rata de învățare pentru optimizator
LOSS_FN = nn.MSELoss() # Funcția de pierdere (Mean Squared Error pentru regresie)

# Inițializăm dataset-urile și dataloader-ele
# Pentru setul de antrenament, `has_labels` este implicit True
train_dataset = TextComplexityDataset(train, bert_tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Pentru setul de test, specificăm explicit `has_labels=False`
test_dataset = TextComplexityDataset(test, bert_tokenizer, has_labels=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # Nu amestecăm setul de test

# Inițializăm modelul și îl mutăm pe dispozitivul (GPU/CPU) detectat
model = BertForRegression(BERT_MODEL_NAME).to(DEVICE)
# Inițializăm optimizatorul
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 6. Loop-ul de Antrenament ---
print("Începe antrenamentul modelului BERT pentru regresie...")

for epoch in range(EPOCHS):
    model.train() # Setează modelul în modul de antrenament
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Train")

    for batch in progress_bar:
        # Mută input-urile și label-urile pe dispozitiv
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE) # 'labels' va exista sigur în batch-urile de train

        optimizer.zero_grad() # Golește gradienții calculați anterior
        predictions = model(input_ids, attention_mask) # Pasul forward: modelul face predicții
        loss = LOSS_FN(predictions, labels) # Calculează pierderea
        loss.backward() # Pasul backward: calculează gradienții
        optimizer.step() # Actualizează greutățile modelului

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

    # --- Evaluare după fiecare epocă (pe setul de antrenament pentru feedback rapid) ---
    model.eval() # Setează modelul în modul de evaluare
    all_preds = []
    all_labels = []
    with torch.no_grad(): # Dezactivează calculul gradientilor
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Train Eval"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            predictions = model(input_ids, attention_mask)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    train_score_r2 = pearson_r2(np.array(all_preds), np.array(all_labels))
    print(f"Epoch {epoch+1} Train Pearson R2 Score: {train_score_r2:.4f}")

print("Antrenament finalizat.")

# --- 7. Generarea predicțiilor pentru submission.csv ---
print("Generare predicții pentru submission.csv cu modelul BERT fine-tuned...")

model.eval() # Asigură-te că modelul este în modul de evaluare
test_preds = []
with torch.no_grad(): # Nu avem nevoie de gradienți pentru predicție
    for batch in tqdm(test_dataloader, desc="Predicting on Test Data"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        # Aici nu mai avem nevoie de `batch['labels']`
        predictions = model(input_ids, attention_mask)
        test_preds.extend(predictions.cpu().numpy())

# Convertim lista de predicții în array NumPy și ne asigurăm că sunt non-negative
preds = np.array(test_preds)
preds = np.maximum(0, preds)

# Creăm fișierul submission.csv
submission = pd.DataFrame({'Id': test.Id.values, 'complexity': preds})
submission.to_csv("submission.csv", index=False)

print("Fișier submission.csv generat cu succes cu modelul BERT fine-tuned.")
print("Acum poți încărca fișierul submission.csv pe Kaggle.")