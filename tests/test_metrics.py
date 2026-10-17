import pytest
from backend.metrics import calculate_bleu, tokenize_sentence, get_ngrams

def test_tokenize_sentence():
    sentence = "A Golden Retriever, jumping in 2026!"
    tokens = tokenize_sentence(sentence)
    assert tokens == ["a", "golden", "retriever", "jumping", "in", "2026"]

def test_get_ngrams():
    tokens = ["a", "dog", "running"]
    unigrams = get_ngrams(tokens, 1)
    bigrams = get_ngrams(tokens, 2)
    assert sum(unigrams.values()) == 3
    assert sum(bigrams.values()) == 2
    assert ("a", "dog") in bigrams

def test_bleu_exact_match():
    candidate = "a dog playing with a red ball"
    references = ["a dog playing with a red ball"]
    scores = calculate_bleu(candidate, references)
    assert scores["bleu_1"] == 1.0
    assert scores["bleu_2"] == 1.0
    assert scores["bleu_3"] == 1.0
    assert scores["bleu_4"] == 1.0
    assert scores["brevity_penalty"] == 1.0

def test_bleu_partial_match():
    candidate = "a dog playing on the grass"
    references = [
        "a dog running on the green grass",
        "a playful dog playing in the field"
    ]
    scores = calculate_bleu(candidate, references)
    assert scores["bleu_1"] > 0.0
    assert scores["bleu_2"] > 0.0
    assert scores["candidate_length"] == 6

def test_bleu_empty_candidate():
    scores = calculate_bleu("", ["some reference"])
    assert scores["bleu_1"] == 0.0
    assert scores["bleu_4"] == 0.0
