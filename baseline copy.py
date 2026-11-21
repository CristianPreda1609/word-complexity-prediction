import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import r2_score, make_scorer
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV # Am adăugat RandomizedSearchCV
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
import lightgbm as lgb

# Asigură-te că nltk.punkt este descărcat
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# ====================
# Citirea datelor
# ====================
BASE_DIR = './date' # Asigură-te că directorul 'date' există și conține fișierele

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

# NOI CARACTERISTICI:
def get_sentence_length_words(sentence):
    # Numărul de cuvinte din propoziție
    return len(word_tokenize(sentence))

def get_word_position_norm(word, sentence):
    # Poziția normalizată a cuvântului în propoziție
    tokens = word_tokenize(sentence)
    try:
        # Găsim prima apariție a cuvântului
        idx = tokens.index(word)
        return idx / len(tokens) if len(tokens) > 0 else 0
    except ValueError:
        # Dacă cuvântul nu e găsit în tokenii propoziției (ex: tokenizare diferită)
        return 0

# ====================
# BERT Setup
# ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert_model.eval()

# FUNCTIA BERT IMBUNATATITA (cu fix pentru eroare)
# Modificăm funcția get_bert_embedding din nou.
@torch.no_grad()
def get_bert_embedding(sentence, word, tokenizer, bert_model, device):
    # Specifică max_length pentru a scăpa de avertismentul de trunchiere
    inputs = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_offsets_mapping=True)

    # Extrage și squeeze input_ids și offsets de la început
    # Aceasta e linia cheie pentru a ne asigura că sunt 1D sau 2D corespunzător
    input_ids = inputs['input_ids'].squeeze(0) # Va fi [seq_len]
    offsets = inputs.pop('offset_mapping').squeeze(0) # Va fi [seq_len, 2]

    inputs = {k: v.to(device) for k, v in inputs.items()} # Treci inputs la GPU după ce ai extras input_ids și offsets
    outputs = bert_model(**inputs)
    hidden_states = outputs.last_hidden_state.squeeze(0) # Va fi [seq_len, hidden_dim]

    word_start_char = sentence.find(word)
    if word_start_char == -1:
        return np.zeros(bert_model.config.hidden_size)

    word_end_char = word_start_char + len(word)

    word_token_embeddings = []
    # Acum `offsets` și `input_ids` sunt ambele 1D (sau 2D unde e cazul)
    for i, (start_offset, end_offset) in enumerate(offsets):
        # Folosește direct `input_ids` care a fost deja squeezed
        token_id = input_ids[i].item()

        if token_id in tokenizer.all_special_ids:
            continue

        if max(start_offset, word_start_char) < min(end_offset, word_end_char):
            word_token_embeddings.append(hidden_states[i])

    if word_token_embeddings:
        return torch.stack(word_token_embeddings).mean(dim=0).cpu().numpy()
    else:
        return np.zeros(bert_model.config.hidden_size)

# ====================
# Combinare toate caracteristicile
# ====================
# FUNCTIA FEATURIZE_ROW MODIFICATA PENTRU NOILE CARACTERISTICI SI BERT IMBUNATATIT
def featurize_row(row, tokenizer, bert_model, device):
    word = row['token']
    sentence = row['sentence']
    features = [
        get_wd_len(word),
        get_frequency(word),
        get_num_vowels(word),
        get_num_consonants(word),
        get_num_syllables(word),
        is_title_or_entity(word),
        get_sentence_length_words(sentence), # Noua caracteristică
        get_word_position_norm(word, sentence) # Noua caracteristică
    ]
    bert_embed = get_bert_embedding(sentence, word, tokenizer, bert_model, device)
    return np.concatenate([features, bert_embed])

# FUNCTIA FEATURIZE_DF MODIFICATA PENTRU A PASA PARAMETRII BERT
def featurize_df(df, tokenizer, bert_model, device):
    return np.vstack([featurize_row(row, tokenizer, bert_model, device) for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing DataFrame")])

# ====================
# Extragem date
# ====================
print("Extragere caracteristici pentru setul de antrenament...")
# APELURILE FEATURIZE_DF MODIFICATE PENTRU A PASA PARAMETRII BERT
X_train = featurize_df(train, tokenizer, bert_model, DEVICE)
y_train = train.score.values
print("Extragere caracteristici pentru setul de testare...")
X_test = featurize_df(test, tokenizer, bert_model, DEVICE)

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

custom_scorer = make_scorer(pearson_r2, greater_is_better=True)

model = lgb.LGBMRegressor(random_state=42)

# GRILA DE PARAMETRI EXTINSA PENTRU LIGHTGBM:
param_grid = {
    'n_estimators': [100, 200, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40, 50],
    'max_depth': [-1, 7, 10, 15],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.0, 0.1, 0.5, 1.0],
}

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

print("Începe Grid Search pentru LightGBM Regressor cu parametri extinsi...")

# Recomandare: Folosește RandomizedSearchCV pentru o explorare mai rapidă a grilei mari
# Dacă ai mult timp la dispoziție și vrei o căutare exhaustivă, poți folosi GridSearchCV
grid_search = RandomizedSearchCV(estimator=model,
                                 param_distributions=param_grid,
                                 n_iter=100, # Numărul de combinații de testat. Crește-l pentru o explorare mai amplă.
                                 scoring=custom_scorer,
                                 cv=cv_strategy,
                                 n_jobs=-1, # Utilizează toate nucleele disponibile
                                 verbose=2, # Afișează progresul
                                 random_state=42)

# Dacă vrei GridSearchCV (mai lent, dar exhaustiv):
# grid_search = GridSearchCV(estimator=model,
#                            param_grid=param_grid,
#                            scoring=custom_scorer,
#                            cv=cv_strategy,
#                            n_jobs=-1,
#                            verbose=2)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"\nCel mai bun LightGBM model: {grid_search.best_params_}")
print(f"Cel mai bun scor Cross-Validation (pearson_r2) cu LightGBM: {grid_search.best_score_}")

train_preds = best_model.predict(X_train_scaled)
print("Scor train (cu cel mai bun model LightGBM):", pearson_r2(train_preds, y_train))

# ====================
# Predictie și Submission
# ====================
print("Generare predicții pentru submission.csv...")
preds = best_model.predict(X_test_scaled)

preds = np.maximum(0, preds) # Asigură-te că toate predicțiile sunt non-negative
submission = pd.DataFrame({'Id': test.Id.values, 'complexity': preds})
submission.to_csv("submission.csv", index=False)

print("Fișier submission.csv generat cu succes cu LightGBM optimizat.")
print("Acum poți încărca fișierul submission.csv pe Kaggle.")