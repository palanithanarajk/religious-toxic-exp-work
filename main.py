import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    jaccard_score, hamming_loss, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import warnings

# --- Configuration ---
MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LENGTH = 256
EPOCHS = 10
PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 2e-5
METRIC_FOR_BEST_MODEL = 'eval_micro_f1'
VALIDATION_SPLIT_SIZE = 0.2
SEED = 42

set_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

warnings.filterwarnings("ignore")

RELIGION_COLS = ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion']
TEXT_COL = 'comment_text'

# --- Data Preprocessing ---
def filter_and_binarize(df, target_cols, min_toxicity=0.5):
    valid_target_cols = [col for col in target_cols if col in df.columns]
    if not valid_target_cols:
        return pd.DataFrame()
    df_filled = df.fillna({col: 0 for col in valid_target_cols})
    toxic_mask = df_filled[valid_target_cols].ge(min_toxicity).any(axis=1)
    cols_to_select = [TEXT_COL] + valid_target_cols
    filtered = df_filled.loc[toxic_mask, cols_to_select].copy()
    filtered[valid_target_cols] = (filtered[valid_target_cols] >= min_toxicity).astype(int)
    filtered.dropna(subset=[TEXT_COL], inplace=True)
    return filtered.reset_index(drop=True)

def balance_multilabel_df(df, target_cols):
    from sklearn.utils import resample
    dfs = []
    max_count = df[target_cols].sum().max()
    for col in target_cols:
        df_pos = df[df[col] == 1]
        df_neg = df[df[col] == 0]
        df_pos_upsampled = resample(df_pos, replace=True, n_samples=max_count, random_state=SEED)
        df_neg_downsampled = resample(df_neg, replace=False, n_samples=max_count, random_state=SEED)
        dfs.append(pd.concat([df_pos_upsampled, df_neg_downsampled]))
    return pd.concat(dfs).drop_duplicates().reset_index(drop=True)

# --- Dataset ---
class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = np.array(texts)
        self.labels = np.array(labels).astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# --- Model Components ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.epsilon = 1e-6

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.epsilon, 1.0 - self.epsilon)
        pt = torch.where(targets == 1, probs, 1 - probs)
        bce_loss = -torch.log(pt)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class HierarchicalAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.1, num_granularity_levels=2):
        super().__init__()
        self.num_granularity_levels = num_granularity_levels
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads,
                                  dropout=attention_dropout, batch_first=True)
            for _ in range(num_granularity_levels)
        ])
        self.output_projection = nn.Linear(hidden_size * num_granularity_levels, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()
        granularity_outputs = []
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for level in range(self.num_granularity_levels):
            window_size = 2 ** level
            num_chunks = seq_length // window_size
            reshaped_states = hidden_states[:, :num_chunks*window_size, :].reshape(batch_size * num_chunks, window_size, hidden_size)
            level_mask = key_padding_mask[:, :num_chunks*window_size].reshape(batch_size * num_chunks, window_size) if key_padding_mask is not None else None
            attn_output, _ = self.attention_layers[level](reshaped_states, reshaped_states, reshaped_states, key_padding_mask=level_mask)
            granularity_outputs.append(attn_output.reshape(batch_size, num_chunks*window_size, hidden_size))
        combined = torch.cat(granularity_outputs, dim=-1)
        return self.layer_norm(self.dropout(self.output_projection(combined)) + hidden_states)

class AdaptivePoolingClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size * 4, hidden_size)
        self.dense2 = nn.Linear(hidden_size, num_labels)
        self.attn_proj = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output, attention_mask=None):
        cls_output = sequence_output[:, 0]
        mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else None
        mean_output = torch.sum(sequence_output * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        max_output = torch.max(sequence_output + (1.0 - mask) * -1e9, dim=1)[0]
        scores = self.attn_proj(sequence_output).squeeze(-1)
        if attention_mask is not None: scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights.unsqueeze(1), sequence_output).squeeze(1)
        pooled = torch.cat([cls_output, mean_output, max_output, attn_output], dim=-1)
        return self.dense2(F.dropout(F.gelu(self.dense1(pooled)), 0.1))

class CustomDebertaV3ForMultilabel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = FocalLoss()
        self.hierarchical_attention = HierarchicalAttentionLayer(config.hidden_size, config.num_attention_heads)
        self.adaptive_classifier = AdaptivePoolingClassifier(config.hidden_size, config.num_labels)
        if hasattr(self, 'classifier'): del self.classifier
        if hasattr(self, 'pooler'): del self.pooler

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.deberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        enhanced = self.hierarchical_attention(sequence_output, attention_mask=attention_mask)
        logits = self.adaptive_classifier(enhanced, attention_mask=attention_mask)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

# --- Metrics & Tuning ---
def compute_metrics_eval(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    true = (labels > 0).astype(int)
    return {
        'accuracy': accuracy_score(true, preds),
        'micro_f1': f1_score(true, preds, average='micro', zero_division=0),
        'macro_f1': f1_score(true, preds, average='macro', zero_division=0),
    }

def find_best_thresholds(logits, labels):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    true = (labels > 0).astype(int)
    best_thresh, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(true, (probs >= t).astype(int), average='micro', zero_division=0)
        if f1 > best_f1: best_f1, best_thresh = f1, t
    return best_thresh

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading data...")
    # Note: User must provide train.csv
    if not os.path.exists("train.csv"):
        print("Error: train.csv not found. Please place it in the directory.")
        exit()
    
    df_original = pd.read_csv("train.csv")
    religion_df = filter_and_binarize(df_original, RELIGION_COLS)
    religion_df = balance_multilabel_df(religion_df, RELIGION_COLS)
    print(f"Total samples: {len(religion_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_dataset = ToxicCommentsDataset(religion_df[TEXT_COL].values, religion_df[RELIGION_COLS].values, tokenizer, MAX_LENGTH)
    
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=VALIDATION_SPLIT_SIZE, random_state=SEED)
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    model = CustomDebertaV3ForMultilabel.from_pretrained(MODEL_NAME, num_labels=len(RELIGION_COLS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        compute_metrics=compute_metrics_eval,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating and tuning threshold...")
    preds = trainer.predict(val_subset)
    best_t = find_best_thresholds(preds.predictions, religion_df.iloc[val_idx][RELIGION_COLS].values)
    print(f"Best threshold: {best_t:.4f}")

    # Final Metrics
    probs = torch.sigmoid(torch.tensor(preds.predictions)).numpy()
    final_preds = (probs >= best_t).astype(int)
    true = (religion_df.iloc[val_idx][RELIGION_COLS].values > 0).astype(int)
    print(classification_report(true, final_preds, target_names=RELIGION_COLS))
