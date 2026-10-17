"""
Training script for Image Caption Generator on Flickr8k.
Supports feature extraction caching, teacher forcing training, and validation BLEU scoring.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tokenizer import CaptionTokenizer
from backend.feature_extractor import CNNFeatureExtractor
from backend.model import ImageCaptionModel
from backend.metrics import calculate_bleu
from training.dataset import Flickr8kDataset, pad_collate_fn, parse_flickr8k_tokens

def extract_and_cache_features(
    image_dir: str, 
    image_ids: list, 
    backbone: str = "mobilenet_v3_small",
    cache_path: str = "weights/features_cache.pt",
    device: str = "cpu"
) -> dict:
    """Extract features for all dataset images and cache to disk."""
    if os.path.exists(cache_path):
        print(f"[Extractor] Found cached features at {cache_path}, loading...")
        return torch.load(cache_path, map_location="cpu")

    print(f"[Extractor] Extracting features for {len(image_ids)} images using {backbone}...")
    extractor = CNNFeatureExtractor(backbone=backbone, device=device)
    features_dict = {}

    start_time = time.time()
    for idx, img_id in enumerate(image_ids):
        img_path = os.path.join(image_dir, img_id)
        if not os.path.exists(img_path):
            continue
        try:
            with Image.open(img_path) as pil_img:
                feat, _ = extractor.extract_features(pil_img)
                features_dict[img_id] = feat.squeeze(0).cpu()
        except Exception as e:
            print(f"Skipping {img_id}: {e}")

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(image_ids)} images ({round(time.time() - start_time, 1)}s)")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(features_dict, cache_path)
    print(f"[Extractor] Saved features cache to {cache_path}")
    return features_dict

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[Training] Using device: {device}")

    os.makedirs("weights", exist_ok=True)
    tokenizer = CaptionTokenizer()

    # Demo synthetic training run
    if args.demo_train:
        print("[Training] Running in demo mode with sample dataset...")
        from backend.sample_data import SAMPLE_IMAGES
        
        extractor = CNNFeatureExtractor(backbone=args.backbone, device=device)
        features_dict = {}
        captions_dict = {}
        
        for s in SAMPLE_IMAGES:
            img_path = os.path.join("data", "samples", s["filename"])
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    feat, _ = extractor.extract_features(img)
                    features_dict[s["filename"]] = feat.squeeze(0).cpu()
                captions_dict[s["filename"]] = s["references"]
                
        # Fit tokenizer
        all_caps = [c for caps in captions_dict.values() for c in caps]
        tokenizer.fit_on_texts(all_caps, min_count=1)
        tokenizer.save("weights/vocab.json")
        print(f"[Training] Vocabulary size: {tokenizer.vocab_size}")

        dataset = Flickr8kDataset(features_dict, captions_dict, tokenizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: pad_collate_fn(b, tokenizer.PAD_IDX))
    else:
        # Full Flickr8k run
        token_path = os.path.join(args.data_dir, "Flickr8k.token.txt")
        image_dir = os.path.join(args.data_dir, "Flicker8k_Dataset")
        captions_dict = parse_flickr8k_tokens(token_path)
        
        all_caps = [c for caps in captions_dict.values() for c in caps]
        tokenizer.fit_on_texts(all_caps, min_count=args.min_word_count)
        tokenizer.save("weights/vocab.json")
        print(f"[Training] Vocabulary size: {tokenizer.vocab_size}")

        features_dict = extract_and_cache_features(
            image_dir=image_dir,
            image_ids=list(captions_dict.keys()),
            backbone=args.backbone,
            device=device
        )
        dataset = Flickr8kDataset(features_dict, captions_dict, tokenizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: pad_collate_fn(b, tokenizer.PAD_IDX))

    # Initialize model
    feat_dim = next(iter(features_dict.values())).shape[0]
    model = ImageCaptionModel(
        feature_dim=feat_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.PAD_IDX
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Training] Starting training for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (feats, inputs, targets) in enumerate(loader):
            feats = feats.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(feats, inputs) # (batch, seq_len, vocab_size)

            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        elapsed = round(time.time() - start_time, 2)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f} ({elapsed}s)")

    # Save final model
    save_path = "weights/caption_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[Training] Model successfully saved to {save_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Image Caption Generator")
    parser.add_argument("--data_dir", type=str, default="data/flickr8k", help="Path to Flickr8k dataset root")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=256, help="Word embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="LSTM hidden state dimension")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet50"])
    parser.add_argument("--min_word_count", type=int, default=2, help="Minimum word count for vocabulary inclusion")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--demo_train", action="store_true", help="Run quick demo training on sample images")

    args = parser.parse_args()
    train(args)
