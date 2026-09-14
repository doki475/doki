"""Core plagiarism detection algorithms.

The algorithm uses a weighted combination of multiple similarity measures
to compute the plagiarism rate between two texts:

1. **Character n-gram Jaccard similarity** (50%):
   - Bigram Jaccard (60% of this component)
   - Trigram Jaccard (40% of this component)
   - Captures exact substring overlap at character level

2. **Character frequency cosine similarity** (30%):
   - Treats each text as a bag of characters
   - Captures overall character distribution similarity

3. **Word Jaccard similarity via jieba** (20%, fallback to skip-gram):
   - Segments text into words for semantic-level comparison
   - Falls back to skip-gram Jaccard if jieba is unavailable
"""

from typing import Dict, Optional, Set

from .preprocessor import extract_char_ngrams, normalize_text

try:
    import jieba
except ImportError:
    jieba = None


# ---------- similarity primitives ----------

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity between two sets: |A ∩ B| / |A ∪ B|."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    """Cosine similarity between two frequency vectors."""
    if not vec1 or not vec2:
        return 0.0

    all_keys = set(vec1) | set(vec2)
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
    norm1 = sum(v * v for v in vec1.values()) ** 0.5
    norm2 = sum(v * v for v in vec2.values()) ** 0.5

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


# ---------- feature extraction ----------

def char_frequency_vector(text: str) -> Dict[str, int]:
    """Build character frequency vector from text."""
    freq: Dict[str, int] = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    return freq


def _compute_word_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity using jieba if available."""
    if jieba is not None:
        words1 = set(jieba.cut(text1, cut_all=False))
        words2 = set(jieba.cut(text2, cut_all=False))
        return jaccard_similarity(words1, words2)

    # Fallback: use 4-gram Jaccard as a rough word-level proxy.
    fourgram_j = jaccard_similarity(
        extract_char_ngrams(text1, 4),
        extract_char_ngrams(text2, 4),
    )
    # Also use skip-grams to capture reordering.
    skip1 = set()
    skip2 = set()
    if len(text1) >= 4:
        for i in range(len(text1) - 3):
            skip1.add(text1[i] + text1[i + 2] + text1[i + 3])
    if len(text2) >= 4:
        for i in range(len(text2) - 3):
            skip2.add(text2[i] + text2[i + 2] + text2[i + 3])
    skip_j = jaccard_similarity(skip1, skip2)
    return 0.6 * fourgram_j + 0.4 * skip_j


# ---------- main similarity computation ----------

def calculate_similarity(text1: Optional[str], text2: Optional[str]) -> float:
    """Calculate the plagiarism rate between two texts.

    The result is a float in [0, 1] where:
    - 1.0 means identical texts
    - 0.0 means completely different texts
    - Values in between represent partial similarity

    Args:
        text1: Original text content
        text2: Plagiarized text content

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Handle None
    if text1 is None:
        text1 = ""
    if text2 is None:
        text2 = ""

    # Normalize texts
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    # Edge cases
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    # ----- 1. Character n-gram Jaccard (50% weight) -----
    bigram_set1 = extract_char_ngrams(text1, 2)
    bigram_set2 = extract_char_ngrams(text2, 2)
    trigram_set1 = extract_char_ngrams(text1, 3)
    trigram_set2 = extract_char_ngrams(text2, 3)

    bigram_j = jaccard_similarity(bigram_set1, bigram_set2)
    trigram_j = jaccard_similarity(trigram_set1, trigram_set2)

    ngram_similarity = 0.6 * bigram_j + 0.4 * trigram_j

    # ----- 2. Character frequency cosine similarity (30% weight) -----
    vec1 = char_frequency_vector(text1)
    vec2 = char_frequency_vector(text2)
    char_cosine = cosine_similarity(vec1, vec2)

    # ----- 3. Word-level similarity (20% weight) -----
    word_sim = _compute_word_similarity(text1, text2)

    # Weighted combination
    similarity = 0.50 * ngram_similarity + 0.30 * char_cosine + 0.20 * word_sim

    # Clamp to valid range
    return max(0.0, min(1.0, similarity))
