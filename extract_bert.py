import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import time
import os

print("Loading data...")
# Read chief complaints
cc_df = pd.read_csv('chief_complaints.csv')

# Only process unique sentences to save massive amounts of time
unique_text = cc_df['chief_complaint_raw'].fillna('unknown').unique()
print(f"Total rows: {len(cc_df)}, Unique complaints: {len(unique_text)}")

print("Loading Bio_ClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Using device: {device}")
print("Starting extraction (this might take a few moments)...")

batch_size = 64
unique_embeddings = []

start_time = time.time()
with torch.no_grad():
    for i in range(0, len(unique_text), batch_size):
        batch_text = unique_text[i:i+batch_size].tolist()
        inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        # Use [CLS] token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        unique_embeddings.append(cls_embeddings)
        
        if (i % 640 == 0) and i > 0:
            print(f"Processed {i}/{len(unique_text)}...")

unique_embeddings = np.vstack(unique_embeddings)
print(f"Extraction complete in {time.time() - start_time:.1f}s.")

# Map back to original dataframe
print("Mapping to all patients...")
feat_cols = [f'bert_vec_{i}' for i in range(768)]
emb_df = pd.DataFrame(unique_embeddings, columns=feat_cols)
emb_df['chief_complaint_raw'] = unique_text

# Merge
cc_df['chief_complaint_raw'] = cc_df['chief_complaint_raw'].fillna('unknown')
final_df = cc_df.merge(emb_df, on='chief_complaint_raw', how='left')

# Save to pickle which preserves types and is fast to load
out_cols = ['patient_id'] + feat_cols
final_df[out_cols].to_pickle('cc_bert_features.pkl')

print("Saved 768-D features to cc_bert_features.pkl successfully!")
