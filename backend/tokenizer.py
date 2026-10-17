"""
Tokenizer and Vocabulary builder for Image Captioning.
"""
import re
import json
import os
from typing import List, Dict, Optional

class CaptionTokenizer:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"
    
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3

    def __init__(self):
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.START_TOKEN: self.START_IDX,
            self.END_TOKEN: self.END_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.START_IDX: self.START_TOKEN,
            self.END_IDX: self.END_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }
        self.word_counts: Dict[str, int] = {}
        self._init_default_vocabulary()

    def _clean_text(self, text: str) -> List[str]:
        text = text.lower()
        # Remove punctuation except letters and numbers
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return [w for w in text.split() if w]

    def _init_default_vocabulary(self):
        """Seed a rich, curated baseline vocabulary for Flickr-style image captioning."""
        seed_words = [
            # Articles & prepositions
            "a", "an", "the", "in", "on", "at", "by", "with", "from", "to", "into", "through",
            "over", "under", "above", "below", "beside", "near", "across", "along", "behind",
            "front", "of", "and", "or", "while", "during", "as", "is", "are", "was", "were",
            # Common subjects / people
            "dog", "cat", "man", "woman", "boy", "girl", "child", "baby", "person", "people",
            "athlete", "surfer", "hiker", "cyclist", "rider", "player", "crowd", "toddler",
            # Animals
            "retriever", "golden", "puppy", "kitten", "bird", "horse", "cow", "sheep", "duck",
            # Environment / locations
            "park", "grass", "field", "mountain", "beach", "ocean", "sea", "water", "river",
            "lake", "road", "street", "sidewalk", "path", "trail", "forest", "trees", "woods",
            "city", "room", "playroom", "kitchen", "rug", "carpet", "ground", "dirt", "snow",
            "sky", "cloud", "sunset", "sunrise", "hill", "peak", "summit", "rock", "rocks",
            # Objects & gear
            "frisbee", "ball", "disc", "bicycle", "bike", "helmet", "backpack", "jacket",
            "shirt", "pants", "shorts", "shoes", "blocks", "toy", "toys", "basket", "flowers",
            "leaves", "trees", "table", "chair", "surfboard", "wave", "camera", "stick",
            # Actions / verbs (present participle, base, etc.)
            "jumping", "jumps", "jump", "running", "runs", "run", "playing", "plays", "play",
            "walking", "walks", "walk", "sitting", "sits", "sit", "standing", "stands", "stand",
            "riding", "rides", "ride", "catching", "catches", "catch", "holding", "holds", "hold",
            "stacking", "stacks", "stack", "climbing", "climbs", "climb", "looking", "looks",
            "watching", "watches", "swimming", "flying", "smiling", "smiles", "laughing",
            "wearing", "wears", "carrying", "carries", "eating", "preparing", "cooking",
            # Colors & attributes
            "red", "blue", "green", "yellow", "orange", "pink", "purple", "black", "white",
            "brown", "golden", "colorful", "bright", "sunny", "autumn", "fall", "tall",
            "little", "small", "big", "large", "young", "old", "happy", "joyful", "rocky",
            "wooden", "striped", "lush", "misty", "dramatic", "warm", "clear", "outstretched",
            "arms", "hands", "midair", "air", "high"
        ]
        for word in seed_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def fit_on_texts(self, texts: List[str], min_count: int = 2):
        """Build vocabulary from a list of caption strings."""
        for text in texts:
            words = self._clean_text(text)
            for w in words:
                self.word_counts[w] = self.word_counts.get(w, 0) + 1

        for word, count in self.word_counts.items():
            if count >= min_count and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert a sentence string to list of token IDs."""
        words = self._clean_text(text)
        seq = [self.word2idx.get(w, self.UNK_IDX) for w in words]
        if add_special_tokens:
            seq = [self.START_IDX] + seq + [self.END_IDX]
        return seq

    def decode(self, token_ids: List[int], strip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back into a readable sentence."""
        words = []
        for idx in token_ids:
            if strip_special_tokens and idx in (self.PAD_IDX, self.START_IDX, self.END_IDX):
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "word_counts": self.word_counts
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.word_counts = data.get("word_counts", {})
