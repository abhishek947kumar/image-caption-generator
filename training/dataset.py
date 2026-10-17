"""
Flickr8k Dataset Loader and PyTorch Dataset representation.
Supports parsing standard Flickr8k text format, pre-extracting CNN features, and batch collation.
"""
import os
import re
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k Image-Caption pairs.
    Each image has 5 reference captions.
    """
    def __init__(
        self,
        features_dict: Dict[str, torch.Tensor],
        captions_dict: Dict[str, List[str]],
        tokenizer,
        max_length: int = 35
    ):
        self.features_dict = features_dict
        self.captions_dict = captions_dict
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Flatten into (image_id, single_caption) pairs
        self.samples: List[Tuple[str, str]] = []
        for img_id, caps in captions_dict.items():
            if img_id in features_dict:
                for c in caps:
                    self.samples.append((img_id, c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_id, caption = self.samples[idx]
        feature = self.features_dict[img_id]  # (feature_dim,)

        token_ids = self.tokenizer.encode(caption, add_special_tokens=True)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Truncate if exceeds max length
        if len(token_tensor) > self.max_length:
            token_tensor = token_tensor[:self.max_length]

        # Input tokens: [<start>, w1, w2, ..., wn]
        # Target tokens: [w1, w2, ..., wn, <end>]
        inputs = token_tensor[:-1]
        targets = token_tensor[1:]

        return feature, inputs, targets

def pad_collate_fn(batch, pad_idx: int = 0):
    """Custom collate function to pad variable length token sequences in a batch."""
    features, inputs, targets = zip(*batch)
    
    features_tensor = torch.stack(features, dim=0)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    return features_tensor, padded_inputs, padded_targets

def parse_flickr8k_tokens(token_file_path: str) -> Dict[str, List[str]]:
    """
    Parse Flickr8k.token.txt format:
    Example: 1000268201_693b08cb0e.jpg#0   A child in a pink dress is climbing...
    """
    captions_dict = {}
    if not os.path.exists(token_file_path):
        raise FileNotFoundError(f"Flickr8k token file not found: {token_file_path}")

    with open(token_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(" ", 1)
            if len(parts) >= 2:
                img_part, cap = parts[0], parts[1]
                # Strip #0, #1 etc suffix
                img_id = img_part.split("#")[0].strip()
                if img_id not in captions_dict:
                    captions_dict[img_id] = []
                captions_dict[img_id].append(cap.strip())
                
    return captions_dict
