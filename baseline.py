import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import r2_score, make_scorer
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
import lightgbm as lgb # NOU: Importăm LightGBM

# Asigură-te că nltk.punkt este descărcat
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# ====================
# Citirea datelor
# ====================
BASE_DIR = './date' # Asigură-te că directorul 'date' există și conține fișierele

# Aici ar trebui să ai fișierele train.csv și test.csv în directorul ./date
try:
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
except FileNotFoundError as e:
    print(f"Eroare la citirea fișierelor: {e}. Asigură-te că train.csv și test.csv sunt în directorul '{BASE_DIR}'.")
    exit() # Ieșim dacă fișierele nu sunt găsite

# ====================
# Tokenizare pentru frecvență
# ====================
from nltk.tokenize import word_tokenize

all_tokens = []
for sentence in tqdm(list(train.sentence.values) + list(test.sentence.values), desc="Tokenizing all sentences"):
    all_tokens.extend(word_tokenize(sentence))

counts = Counter(all_tokens)

def get_frequency(word):
    return counts.get(word, 0)

# ====================
# Funcții pentru extragere de caracteristici simple
# ====================
def get_wd_len(word):
    return len(word)

def get_num_vowels(word):
    vowels = "aeiouAEIOUăĂîÎâÂ"
    return sum(1 for char in word if char in vowels)

def get_num_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZșȘțȚ"
    return sum(1 for char in word if char in consonants)

def get_num_syllables(word):
    word = word.lower()
    vowels = "aeiouăîâ"
    num_syllables = 0
    in_vowel_group = False
    for char in word:
        if char in vowels:
            if not in_vowel_group:
                num_syllables += 1
                in_vowel_group = True
        else:
            in_vowel_group = False
    return max(1, num_syllables)

def is_title_or_entity(word):
    return 1 if word and word[0].isupper() else 0

# ====================
# BERT Setup
# ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert_model.eval()

@torch.no_grad()
def get_bert_embedding(sentence, word):
    tokens = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    outputs = bert_model(**tokens)
    hidden_states = outputs.last_hidden_state.squeeze(0)  # shape: [seq_len, hidden_dim]
    input_ids = tokens['input_ids'].squeeze(0)

    target_tokens = tokenizer.tokenize(word)
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    for idx in range(1, len(input_ids) - len(target_ids)):
        if torch.equal(input_ids[idx:idx+len(target_ids)], torch.tensor(target_ids).to(DEVICE)):
            emb = hidden_states[idx:idx+len(target_ids)].mean(dim=0).cpu().numpy()
            return emb

    return np.zeros(bert_model.config.hidden_size)

# ====================
# Combinare toate caracteristicile
# ====================
def featurize_row(row):
    word = row['token']
    sentence = row['sentence']
    features = [
        get_wd_len(word),
        get_frequency(word),
        get_num_vowels(word),
        get_num_consonants(word),
        get_num_syllables(word),
        is_title_or_entity(word)
    ]
    bert_embed = get_bert_embedding(sentence, word)
    return np.concatenate([features, bert_embed])

def featurize_df(df):
    return np.vstack([featurize_row(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing DataFrame")])

# ====================
# Extragem date
# ====================
print("Extragere caracteristici pentru setul de antrenament...")
X_train = featurize_df(train)
y_train = train.score.values
print("Extragere caracteristici pentru setul de testare...")
X_test = featurize_df(test)

# ====================
# Scalarea Caracteristicilor
# ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Caracteristicile au fost scalate.")

# ====================
# Model + evaluare
# ====================
def pearson_r2(preds, y_true):
    r2 = max(0, r2_score(y_true, preds))
    pears = pearsonr(y_true, preds)[0]
    pears = np.nan_to_num(pears)
    return (abs(pears) + r2) / 2

# Creăm un scorer personalizat pentru GridSearchCV
custom_scorer = make_scorer(pearson_r2, greater_is_better=True)

# NOU: Modelul este LightGBM Regressor
model = lgb.LGBMRegressor(random_state=42) # Adăugăm random_state pentru reproductibilitate

# NOU: Definirea parametrilor pentru GridSearchCV pentru LightGBM
# Am ales un set de parametri care ar trebui să ofere un bun echilibru între performanță și timp de rulare
param_grid = {
    'n_estimators': [100, 200],         # Numărul de arbori
    'learning_rate': [0.05, 0.1],      # Mărimea pasului de învățare
    'num_leaves': [20, 31],            # Numărul maxim de frunze într-un arbore (complexitatea arborelui)
    'max_depth': [-1, 7],              # Adâncimea maximă a arborelui (-1 = fără limită, poate duce la overfitting)
    'reg_alpha': [0.1, 0.5],           # Regularizare L1
    'reg_lambda': [0.1, 0.5]           # Regularizare L2
}

# Setup pentru Cross-Validation
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV pentru a găsi cel mai bun model LightGBM
print("Începe Grid Search pentru LightGBM Regressor...")
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring=custom_scorer,
                           cv=cv_strategy,
                           n_jobs=-1, # Utilizează toate nucleele disponibile
                           verbose=2) # Afișează progresul

grid_search.fit(X_train_scaled, y_train)

# Obținem cel mai bun estimator și scor
best_model = grid_search.best_estimator_
print(f"\nCel mai bun LightGBM model: {grid_search.best_params_}")
print(f"Cel mai bun scor Cross-Validation (pearson_r2) cu LightGBM: {grid_search.best_score_}")

# Evaluare pe setul de antrenament cu cel mai bun model
train_preds = best_model.predict(X_train_scaled)
print("Scor train (cu cel mai bun model LightGBM):", pearson_r2(train_preds, y_train))

# ====================
# Predictie și Submission
# ====================
print("Generare predicții pentru submission.csv...")
preds = best_model.predict(X_test_scaled)

# Asigură-te că toate predicțiile sunt non-negative
preds = np.maximum(0, preds)
submission = pd.DataFrame({'Id': test.Id.values, 'complexity': preds})
submission.to_csv("submission.csv", index=False)

print("Fișier submission.csv generat cu succes cu LightGBM optimizat.")
print("Acum poți încărca fișierul submission.csv pe Kaggle.")