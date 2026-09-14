"""Tests for the plagiarism checker package."""

import os
import tempfile
import unittest

from checker.algorithm import (
    char_frequency_vector,
    calculate_similarity,
    jaccard_similarity,
    cosine_similarity,
)
from checker.preprocessor import (
    extract_char_ngrams,
    extract_skip_ngrams,
    normalize_text,
)
from main import read_file


class TestPreprocessor(unittest.TestCase):
    """Tests for text preprocessing."""

    def test_normalize_text_removes_extra_whitespace(self):
        """Whitespace collapsing: multiple spaces → single space."""
        result = normalize_text("hello    world   test")
        self.assertEqual(result, "hello world test")

    def test_normalize_text_handles_newlines(self):
        """Newline handling: newlines become spaces."""
        result = normalize_text("line1\nline2\n\nline3")
        self.assertEqual(result, "line1 line2 line3")

    def test_normalize_text_normalizes_chinese_punctuation(self):
        """Chinese punctuation is normalized to English equivalents."""
        result = normalize_text("你好，世界。测试！")
        self.assertIn(',', result)
        self.assertIn('.', result)
        self.assertIn('!', result)

    def test_extract_char_ngrams_basic(self):
        """Extracting bigrams from 'hello'."""
        result = extract_char_ngrams("hello", 2)
        self.assertEqual(result, {"he", "el", "ll", "lo"})

    def test_extract_char_ngrams_chinese(self):
        """Extracting bigrams from Chinese text."""
        result = extract_char_ngrams("今天天气", 2)
        self.assertEqual(result, {"今天", "天天", "天气"})

    def test_extract_char_ngrams_empty(self):
        """Empty string produces empty set."""
        self.assertEqual(extract_char_ngrams("", 2), set())

    def test_extract_char_ngrams_text_shorter_than_n(self):
        """Text shorter than n produces empty set."""
        self.assertEqual(extract_char_ngrams("ab", 3), set())


class TestJaccardSimilarity(unittest.TestCase):
    """Tests for Jaccard similarity computation."""

    def test_jaccard_identical_sets(self):
        """Identical sets → 1.0."""
        s = {"a", "b", "c"}
        self.assertAlmostEqual(jaccard_similarity(s, s), 1.0)

    def test_jaccard_disjoint_sets(self):
        """Disjoint sets → 0.0."""
        s1 = {"a", "b", "c"}
        s2 = {"d", "e", "f"}
        self.assertAlmostEqual(jaccard_similarity(s1, s2), 0.0)

    def test_jaccard_partial_overlap(self):
        """Sets with 2/3 intersection over 4/3 union → ~0.5."""
        s1 = {"a", "b", "c"}
        s2 = {"b", "c", "d"}
        self.assertAlmostEqual(jaccard_similarity(s1, s2), 0.5)

    def test_jaccard_both_empty(self):
        """Both empty → 1.0 (by convention)."""
        self.assertAlmostEqual(jaccard_similarity(set(), set()), 1.0)

    def test_jaccard_one_empty(self):
        """One empty → 0.0."""
        self.assertAlmostEqual(jaccard_similarity({"a"}, set()), 0.0)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine similarity computation."""

    def test_cosine_identical_vectors(self):
        """Identical vectors → 1.0."""
        v = {"a": 3, "b": 4}
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_cosine_orthogonal_vectors(self):
        """Vectors with no overlapping keys → 0.0."""
        self.assertAlmostEqual(
            cosine_similarity({"a": 1}, {"b": 1}), 0.0
        )

    def test_cosine_partial(self):
        """Partial overlap: (1*2 + 2*1)/(|v1|*|v2|)."""
        v1 = {"a": 1, "b": 2}
        v2 = {"a": 2, "b": 1, "c": 3}
        expected = (1 * 2 + 2 * 1) / (
            (1**2 + 2**2)**0.5 * (2**2 + 1**2 + 3**2)**0.5
        )
        self.assertAlmostEqual(cosine_similarity(v1, v2), expected)

    def test_cosine_empty_vector(self):
        """Empty vector → 0.0."""
        self.assertAlmostEqual(cosine_similarity({}, {"a": 1}), 0.0)


class TestSimilarityIntegration(unittest.TestCase):
    """Integration tests for the full similarity pipeline."""

    def test_identical_texts(self):
        """Two identical texts should have similarity ≈ 1.0."""
        text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        sim = calculate_similarity(text, text)
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_completely_different_texts(self):
        """Two completely different texts should have low similarity."""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "Python是一种广泛使用的解释型高级编程语言。"
        sim = calculate_similarity(text1, text2)
        self.assertLess(sim, 0.3)

    def test_both_empty(self):
        """Both empty → 1.0."""
        self.assertAlmostEqual(
            calculate_similarity("", ""), 1.0
        )

    def test_one_empty_one_not(self):
        """One empty → 0.0."""
        self.assertAlmostEqual(
            calculate_similarity("", "hello world"), 0.0
        )

    def test_minor_modification(self):
        """Small changes should give high similarity."""
        orig = "今天是星期天，天气晴，今天晚上我要去看电影。"
        modified = "今天是周天，天气晴朗，我晚上要去看电影。"
        sim = calculate_similarity(orig, modified)
        self.assertGreater(sim, 0.5)

    def test_major_addition(self):
        """Large additions should reduce similarity."""
        orig = "测试文本"
        modified = "测试文本" + "A" * 1000
        sim = calculate_similarity(orig, modified)
        self.assertLess(sim, 0.5)

    def test_major_deletion(self):
        """Large deletions should reduce similarity."""
        orig = "测试文本" + "A" * 500
        modified = "测试文本"
        sim = calculate_similarity(orig, modified)
        self.assertLess(sim, 0.5)

    def test_half_similar(self):
        """Text with half overlapping content."""
        common = "这是共享的文本部分。"
        unique1 = "这是文本A独有的内容部分一。"
        unique2 = "这是文本B独有的内容部分二。"
        text1 = common + unique1
        text2 = common + unique2
        sim = calculate_similarity(text1, text2)
        self.assertGreater(sim, 0.3)
        self.assertLess(sim, 0.8)

    def test_reordered_content(self):
        """Reordered sentences should still show similarity."""
        text1 = "句子一。句子二。句子三。"
        text2 = "句子三。句子一。句子二。"
        sim = calculate_similarity(text1, text2)
        self.assertGreater(sim, 0.5)

    def test_long_texts_similarity(self):
        """Long texts with similar content."""
        base = "计算机科学是研究计算机及其周围各种现象和规律的科学。"
        text1 = base * 20
        text2 = base * 18 + "人工智能是研究使计算机模拟人类智能行为的科学。" * 2
        sim = calculate_similarity(text1, text2)
        self.assertGreater(sim, 0.7)

    def test_none_inputs(self):
        """None inputs should be handled gracefully."""
        sim = calculate_similarity(None, None)
        self.assertAlmostEqual(sim, 1.0)
        sim = calculate_similarity(None, "hello")
        self.assertAlmostEqual(sim, 0.0)

    def test_special_characters(self):
        """Texts with special characters."""
        text1 = "Hello, 世界! @#$%^&*()"
        text2 = "Hello, 世界! @#$%^&*()"
        sim = calculate_similarity(text1, text2)
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_mixed_language(self):
        """Mixed Chinese and English text."""
        text1 = "Python是一门popular的编程语言，广泛用于AI领域。"
        text2 = "Python是一门流行的编程语言，广泛应用于AI领域。"
        sim = calculate_similarity(text1, text2)
        self.assertGreater(sim, 0.6)

    def test_only_punctuation(self):
        """Texts with only punctuation."""
        sim = calculate_similarity(".,!.,!", ".,!.,!")
        self.assertAlmostEqual(sim, 1.0, places=2)


class TestFileIntegration(unittest.TestCase):
    """Tests that exercise the full file-based pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def _write_file(self, name, content):
        path = os.path.join(self.temp_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_cli_like_pipeline(self):
        """Test reading files and writing output like the CLI would."""
        orig_path = self._write_file(
            "orig.txt",
            "今天是星期天，天气晴，今天晚上我要去看电影。"
        )
        plag_path = self._write_file(
            "plag.txt",
            "今天是周天，天气晴朗，我晚上要去看电影。"
        )
        out_path = os.path.join(self.temp_dir, "ans.txt")

        # Simulate CLI
        orig_text = read_file(orig_path)
        plag_text = read_file(plag_path)
        sim = calculate_similarity(orig_text, plag_text)
        sim = max(0.0, min(1.0, sim))

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"{sim:.2f}")

        with open(out_path, 'r', encoding='utf-8') as f:
            result = f.read().strip()

        # Check format: floating point with 2 decimal places
        value = float(result)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        # Check 2 decimal places
        parts = result.split('.')
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[1]), 2)


class TestSkipNgrams(unittest.TestCase):
    """Tests for skip-gram extraction."""

    def test_extract_skip_ngrams_basic(self):
        """Basic skip-gram extraction."""
        result = extract_skip_ngrams("hello", 2, 1)
        self.assertIsInstance(result, set)
        self.assertGreater(len(result), 0)

    def test_extract_skip_ngrams_empty(self):
        """Empty string produces empty set."""
        self.assertEqual(extract_skip_ngrams("", 2, 1), set())

    def test_extract_skip_ngrams_short_text(self):
        """Text shorter than n produces empty set."""
        self.assertEqual(extract_skip_ngrams("a", 2, 1), set())


class TestReadFile(unittest.TestCase):
    """Tests for file reading utilities."""

    def test_read_file_utf8(self):
        """Reading a UTF-8 encoded file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("测试文本")
            f.flush()
            path = f.name

        try:
            content = read_file(path)
            self.assertEqual(content, "测试文本")
        finally:
            os.unlink(path)

    def test_read_file_gbk(self):
        """Reading a GBK encoded file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='gbk') as f:
            f.write("GBK测试")
            f.flush()
            path = f.name

        try:
            content = read_file(path)
            self.assertEqual(content, "GBK测试")
        finally:
            os.unlink(path)

    def test_read_file_not_found(self):
        """Reading non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_file("/nonexistent/file.txt")


class TestCharFrequencyVector(unittest.TestCase):
    """Tests for character frequency vector."""

    def test_char_frequency_basic(self):
        """Basic character frequency counting."""
        vec = char_frequency_vector("aabbc")
        self.assertEqual(vec, {'a': 2, 'b': 2, 'c': 1})

    def test_char_frequency_empty(self):
        """Empty string produces empty dict."""
        self.assertEqual(char_frequency_vector(""), {})

    def test_char_frequency_chinese(self):
        """Chinese character frequency counting."""
        vec = char_frequency_vector("你好你好")
        self.assertEqual(vec['你'], 2)
        self.assertEqual(vec['好'], 2)


if __name__ == '__main__':
    unittest.main()
