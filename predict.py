import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from torch.utils.data import Dataset


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


model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

test_df = pd.read_csv("./date/test.csv")
test_dataset = ComplexityDataset(test_df, tokenizer, is_test=True)

# Aici incarcam modelul si il apelam pe setul de date de test
trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)
scores = predictions.predictions.squeeze()

output_df = pd.DataFrame({
    "Id": test_df["Id"],
    "complexity": np.clip(scores, 0, 1)
})

output_df.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")
