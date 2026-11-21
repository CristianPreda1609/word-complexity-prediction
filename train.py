import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_df = pd.read_csv("./date/train.csv")
test_df = pd.read_csv("./date/test.csv")

# Impartim setul de date in 2 seturi diferite, unul de 90% pentru antrenare, iar celalalt 10% pentru validare pe date nevazute
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=True)

# Clasa custom pentru PyTorch Dataset
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

            # Evidentiem tokenul tinta in propozitie cu [TGT] ... [TGT]
            marked = sentence.replace(token, f"[TGT] {token} [TGT]", 1)

            # Tokenizarea propozitiei cu cuvantul marcat
            encoded = tokenizer(
                marked,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            item = {
                'input_ids': encoded['input_ids'].squeeze(),        # vectorul de input IDs
                'attention_mask': encoded['attention_mask'].squeeze()  # masca de atentie
            }

            if not is_test:
                # Adaugam scorul de complexitate doar in timpul antrenarii/validarii
                item['labels'] = torch.tensor(row['score'], dtype=torch.float)

            self.samples.append(item)

    def __len__(self):
        # Returneaza numarul total de exemple
        return len(self.samples)

    def __getitem__(self, idx):
        # Returneaza un singur exemplu pregatit pentru model
        return self.samples[idx]


# Incarcarea modelului si tokenizerului
model_name = "readerbench/RoBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(["[TGT]"])  # Adaugarea tokenului special delimitat prin TGT

# Aici incarc modelul preantrenat pentru regresie
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# Aici fac redimensionarea matricii de embedding pentru includerea noului token TGT
model.resize_token_embeddings(len(tokenizer))



from sklearn.model_selection import train_test_split


train_dataset = ComplexityDataset(train_df, tokenizer)
val_dataset = ComplexityDataset(val_df, tokenizer)

train_dataset = ComplexityDataset(train_df, tokenizer)
test_dataset = ComplexityDataset(test_df, tokenizer, is_test=True)
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.squeeze()
    labels = labels.squeeze()

    # Scorul R^2
    r2 = max(0, r2_score(labels, preds))

    # Corelatia Pearson 
    pears = pearsonr(labels, preds)[0]
    pears = np.nan_to_num(pears)  

    # Scorul final reprezinta media dintre cele doua metode de evaluare
    score = (abs(pears) + r2) / 2

    return {
        "pearson": abs(pears),
        "r2": r2,
        "pearson_r2": score
    }

# Setarile de antrenare
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20, # Aici am incercat mai multe epoci, pana acum 20 de epoci au fost cele mai bune ca si durata si scor
    per_device_train_batch_size=16, # Marirea batch-ului nu a ajutat scorul
    learning_rate=2e-5,
    weight_decay=0.01, # Am incercat valor intre 0.05 si 0.001, empiric am ajuns la 0.01 ca fiind o valoare decenta dupa scorul final
    evaluation_strategy="epoch",  # Aici facem o evaluare dupa fiecare epoca sa vad unde ne situam
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="pearson_r2", # Am optat ca metrica de evaluare a celui mai bun model sa fie dupa evaluarea Pearson_R2 
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none"
)

# Trainerul HuggingFace gestioneaza tot procesul de antrenare
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# Antrenarea si salvarea celui mai bun model
trainer.train()

trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
