import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

df = pd.read_csv('./date/train.csv')
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

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

            item = {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            }

            if not is_test and 'score' in row:
                item['labels'] = torch.tensor(row['score'], dtype=torch.float)

            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Load model and tokenizer
model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create dataset
val_dataset = ComplexityDataset(val_df, tokenizer)

# Predict on validation set
trainer = Trainer(model=model)
val_preds = trainer.predict(val_dataset).predictions.squeeze()
val_true = val_df['score'].values
val_preds = np.clip(val_preds, 0, 1)

# Compute metrics
pearson_corr, _ = pearsonr(val_true, val_preds)
spearman_corr, _ = spearmanr(val_true, val_preds)
r2 = r2_score(val_true, val_preds)

print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plot true vs predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=val_true, y=val_preds, alpha=0.6)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('True Complexity Score')
plt.ylabel('Predicted Complexity Score')
plt.title('True vs Predicted Complexity Scores')
plt.grid(True)
plt.savefig('true_vs_predicted.png')
plt.show()

# Residual plot
residuals = val_true - val_preds
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Prediction Error (Residual)')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.savefig('prediction_errors.png')
plt.show()

# Save metrics table
metrics = {
    "Pearson Correlation": [pearson_corr],
    "Spearman Correlation": [spearman_corr],
    "R^2 Score": [r2]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('evaluation_metrics.csv', index=False)
print("✅ Evaluation metrics saved to evaluation_metrics.csv")
