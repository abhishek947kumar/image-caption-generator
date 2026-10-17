"""
Evaluation metrics for Image Captioning.
Implements BLEU-1, BLEU-2, BLEU-3, BLEU-4 with brevity penalty and smoothing.
"""
import math
from collections import Counter
from typing import List, Union

def tokenize_sentence(sentence: str) -> List[str]:
    """Clean and tokenize a sentence into lowercase tokens."""
    import re
    cleaned = re.sub(r"[^\w\s]", "", sentence.lower())
    return [w for w in cleaned.split() if w]

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Generate n-grams and return their count frequencies."""
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

def calculate_bleu(
    candidate: Union[str, List[str]], 
    references: List[Union[str, List[str]]], 
    max_n: int = 4,
    smoothing: bool = True
) -> dict:
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores for a candidate caption
    against one or more reference captions.
    
    Args:
        candidate: Candidate string or list of tokens.
        references: List of reference strings or lists of tokens.
        max_n: Maximum n-gram order (default: 4).
        smoothing: Whether to apply add-epsilon smoothing for 0-count n-grams.
        
    Returns:
        dict: BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores and brevity penalty.
    """
    cand_tokens = tokenize_sentence(candidate) if isinstance(candidate, str) else candidate
    ref_tokens_list = [
        tokenize_sentence(ref) if isinstance(ref, str) else ref
        for ref in references
    ]
    
    cand_len = len(cand_tokens)
    if cand_len == 0:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
    
    # Find closest reference length for Brevity Penalty (BP)
    ref_lens = [len(ref) for ref in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - cand_len), r_len))
    
    # Brevity penalty calculation
    if cand_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - closest_ref_len / cand_len) if cand_len > 0 else 0.0
    
    precisions = []
    
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(cand_tokens, n)
        total_cand_ngrams = sum(cand_ngrams.values())
        
        if total_cand_ngrams == 0:
            precisions.append(0.0)
            continue
            
        # Get max count of each n-gram across all references
        max_ref_counts = {}
        for ref_tokens in ref_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
                
        # Clip candidate n-grams
        clipped_matches = 0
        for ngram, count in cand_ngrams.items():
            clipped_matches += min(count, max_ref_counts.get(ngram, 0))
            
        if clipped_matches == 0:
            if smoothing:
                # Epsilon smoothing for higher order n-grams on short sentences
                p_n = 0.5 / total_cand_ngrams
            else:
                p_n = 0.0
        else:
            p_n = clipped_matches / total_cand_ngrams
            
        precisions.append(p_n)
    
    # Calculate cumulative BLEU scores (BLEU-1, 2, 3, 4)
    scores = {}
    for n in range(1, max_n + 1):
        sub_precisions = precisions[:n]
        if any(p == 0.0 for p in sub_precisions):
            score = 0.0
        else:
            weight = 1.0 / n
            log_sum = sum(weight * math.log(p) for p in sub_precisions)
            score = bp * math.exp(log_sum)
        scores[f"bleu_{n}"] = round(score, 4)
        
    scores["brevity_penalty"] = round(bp, 4)
    scores["candidate_length"] = cand_len
    scores["reference_length"] = closest_ref_len
    scores["precision_1"] = round(precisions[0], 4) if len(precisions) > 0 else 0.0
    scores["precision_2"] = round(precisions[1], 4) if len(precisions) > 1 else 0.0
    scores["precision_3"] = round(precisions[2], 4) if len(precisions) > 2 else 0.0
    scores["precision_4"] = round(precisions[3], 4) if len(precisions) > 3 else 0.0
    
    return scores
