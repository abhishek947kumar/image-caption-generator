# Training Guide: Image Caption Generator on Flickr8k

This guide details how to train the **CNN-LSTM** Image Caption Generator using PyTorch on the benchmark **Flickr8k** dataset.

---

## 1. Flickr8k Dataset Setup

Download the standard Flickr8k dataset (e.g. from Kaggle or official academic mirrors):
1. Images: `Flicker8k_Dataset/` containing 8,092 JPEG images.
2. Captions: `Flickr8k.token.txt` containing 40,460 captions (5 per image).

Place them under `data/flickr8k/`:
```
data/
└── flickr8k/
    ├── Flicker8k_Dataset/
    │   ├── 1000268201_693b08cb0e.jpg
    │   └── ...
    └── Flickr8k.token.txt
```

---

## 2. Fast Training via Offline CNN Feature Caching

Running full forward passes through CNN backbones on every epoch is computationally redundant since the CNN encoder is kept frozen while learning the language decoder. 

`training/train.py` automatically caches feature representations into `weights/features_cache.pt`. Subsequent epochs train purely on the feature vectors at **100x speed**!

---

## 3. Running the Training Pipeline

### Quick Verification Run (Demo on sample images)
```bash
python training/train.py --demo_train --epochs 5 --batch_size 4
```

### Full Flickr8k Dataset Training
```bash
python training/train.py \
  --data_dir data/flickr8k \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.001 \
  --backbone mobilenet_v3_small \
  --embed_dim 256 \
  --hidden_dim 512
```

---

## 4. Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data/flickr8k` | Path to dataset directory containing images and tokens |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `16` | Batch size for gradient descent |
| `--lr` | `0.001` | Adam learning rate |
| `--backbone` | `mobilenet_v3_small` | Pretrained CNN (`mobilenet_v3_small` or `resnet50`) |
| `--embed_dim` | `256` | Word vector representation dimension |
| `--hidden_dim` | `512` | LSTM recurrent state dimension |
| `--min_word_count` | `2` | Minimum frequency threshold to retain a word in vocabulary |
