import json
import torch
from torch.utils.data import Dataset

class EnrichedDataset(Dataset):
    def __init__(self, manifest_path, library_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 1. Load the heavy metadata library into memory (The "Join" Key)
        print(f"Loading metadata library from {library_path}...")
        with open(library_path, 'r') as f:
            self.library = json.load(f)
            
        # 2. Load the lightweight manifest (The "Instruction" List)
        print(f"Loading manifest from {manifest_path}...")
        with open(manifest_path, 'r') as f:
            self.pairs = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 3. Enrich the data by looking up the IDs
        meta_a = self.library.get(pair['drug_1_id'], "")
        meta_b = self.library.get(pair['drug_1_id'], "")
        label = pair['label']

        # 4. Construct the prompt for Qwen3
        # Adjust this template based on your specific task
        text = f"Item A: {meta_a}\nItem B: {meta_b}\nClassification: {label}"
        
        # 5. Tokenize on the fly
        batch = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Flatten the tensors for the DataLoader
        return {k: v.squeeze(0) for k, v in batch.items()}