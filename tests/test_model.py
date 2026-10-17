import pytest
import torch
from backend.model import ImageCaptionModel

def test_model_initialization():
    model = ImageCaptionModel(
        feature_dim=1024,
        embed_dim=128,
        hidden_dim=256,
        vocab_size=500
    )
    counts = model.count_parameters()
    assert counts["total_parameters"] > 0
    assert counts["trainable_parameters"] == counts["total_parameters"]

def test_model_forward_pass_shapes():
    batch_size = 4
    seq_len = 10
    feature_dim = 1024
    embed_dim = 128
    hidden_dim = 256
    vocab_size = 500

    model = ImageCaptionModel(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size
    )

    dummy_features = torch.randn(batch_size, feature_dim)
    dummy_captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(dummy_features, dummy_captions)
    assert logits.shape == (batch_size, seq_len, vocab_size)

def test_model_decode_step():
    model = ImageCaptionModel(
        feature_dim=1024,
        embed_dim=128,
        hidden_dim=256,
        vocab_size=500
    )

    dummy_feature = torch.randn(1, 1024)
    img_embeds = model.image_fc(dummy_feature)
    h_0, c_0 = model.init_hidden_states(img_embeds)

    current_token = torch.tensor([[1]])  # <start>
    logits, (new_h, new_c) = model.decode_step(img_embeds, current_token, (h_0, c_0))

    assert logits.shape == (1, 500)
    assert new_h.shape == (1, 1, 256)
    assert new_c.shape == (1, 1, 256)
