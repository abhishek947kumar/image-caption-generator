import pytest
import os
import tempfile
from backend.tokenizer import CaptionTokenizer

def test_tokenizer_initialization():
    tokenizer = CaptionTokenizer()
    assert tokenizer.PAD_IDX == 0
    assert tokenizer.START_IDX == 1
    assert tokenizer.END_IDX == 2
    assert tokenizer.UNK_IDX == 3
    assert tokenizer.vocab_size > 50

def test_tokenizer_encode_decode():
    tokenizer = CaptionTokenizer()
    sentence = "A golden dog jumping on green grass"
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    
    assert token_ids[0] == tokenizer.START_IDX
    assert token_ids[-1] == tokenizer.END_IDX
    assert len(token_ids) > 2

    decoded = tokenizer.decode(token_ids)
    assert "dog" in decoded
    assert "grass" in decoded

def test_tokenizer_unknown_word():
    tokenizer = CaptionTokenizer()
    sentence = "A supercalifragilisticexpialidocious object"
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    assert tokenizer.UNK_IDX in token_ids

def test_tokenizer_fit_and_save_load():
    tokenizer = CaptionTokenizer()
    corpus = [
        "quantum computer processing information",
        "quantum circuits and qubits"
    ]
    initial_vocab_size = tokenizer.vocab_size
    tokenizer.fit_on_texts(corpus, min_count=1)
    assert tokenizer.vocab_size > initial_vocab_size
    assert "quantum" in tokenizer.word2idx

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "vocab.json")
        tokenizer.save(save_path)
        assert os.path.exists(save_path)

        new_tokenizer = CaptionTokenizer()
        new_tokenizer.load(save_path)
        assert "quantum" in new_tokenizer.word2idx
        assert new_tokenizer.vocab_size == tokenizer.vocab_size
