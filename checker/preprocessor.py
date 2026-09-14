"""Text preprocessing utilities for plagiarism detection."""

import re
from typing import Set


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Steps:
    1. Normalize Chinese punctuation to English equivalents
    2. Collapse whitespace
    3. Remove leading/trailing whitespace
    """
    # Normalize Chinese punctuation to English for consistent comparison
    punctuation_map = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '：': ':', '；': ';', '“': '"', '”': '"',
        '‘': "'", '’': "'", '（': '(', '）': ')',
        '【': '[', '】': ']', '《': '<', '》': '>',
        '、': ',', '～': '~', '　': ' ',
    }
    for chinese_punct, english_punct in punctuation_map.items():
        text = text.replace(chinese_punct, english_punct)

    # Collapse all whitespace (spaces, tabs, newlines) to single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_char_ngrams(text: str, n: int) -> Set[str]:
    """Extract character n-grams from text.

    Args:
        text: Input text
        n: n-gram size (e.g., 2 for bigrams, 3 for trigrams)

    Returns:
        Set of character n-gram strings
    """
    if not text or len(text) < n:
        return set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def extract_skip_ngrams(text: str, n: int, k: int) -> Set[str]:
    """Extract skip-grams from text - n characters with up to k skips.

    This helps capture reworded content where words are reordered.

    Args:
        text: Input text
        n: Number of characters to pick
        k: Maximum gap between consecutive characters

    Returns:
        Set of skip-gram strings
    """
    if not text or len(text) < n:
        return set()

    result = set()
    # Generate n-grams with gaps
    for i in range(len(text) - n + 1):
        for gap in range(min(k + 1, len(text) - i - n + 1)):
            gram = text[i]
            for j in range(1, n):
                pos = i + j + gap
                if pos < len(text):
                    gram += text[pos]
            if len(gram) == n:
                result.add(gram)

    return result
