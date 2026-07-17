"""Ortak veri işlemleri için yalnızca standart kütüphane kullanan testler."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common import (
    SentimentExample,
    binary_classification_metrics,
    build_vocabulary,
    load_examples,
    stratified_split,
    tokenize,
    vectorize_text,
)


class CommonModuleTests(unittest.TestCase):
    def test_turkish_tokenization(self) -> None:
        self.assertEqual(tokenize("İYİ çalışıyor! Çok güzel."), ["iyi", "çalışıyor", "çok", "güzel"])

    def test_stratified_split_keeps_both_classes(self) -> None:
        examples = [
            SentimentExample(f"olumlu {index}", 1) for index in range(6)
        ] + [SentimentExample(f"olumsuz {index}", 0) for index in range(6)]
        train, test = stratified_split(examples, test_ratio=0.25, seed=42)
        self.assertEqual({example.label for example in train}, {0, 1})
        self.assertEqual({example.label for example in test}, {0, 1})
        self.assertEqual(len(train) + len(test), len(examples))

    def test_vocabulary_and_vectorization(self) -> None:
        examples = [
            SentimentExample("çok iyi", 1),
            SentimentExample("çok kötü", 0),
        ]
        vocabulary = build_vocabulary(examples)
        vector = vectorize_text("iyi bilinmeyen", vocabulary)
        self.assertEqual(len(vector), len(vocabulary))
        self.assertAlmostEqual(sum(vector), 1.0)

    def test_binary_metrics(self) -> None:
        metrics = binary_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metrics.true_positive, 2)
        self.assertEqual(metrics.false_positive, 1)
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        self.assertAlmostEqual(metrics.recall, 1.0)

    def test_csv_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("text,label\nharika,1\nkötü,0\nnormal,1\nzayıf,0\n", encoding="utf-8")
            examples = load_examples(path)
            self.assertEqual(len(examples), 4)


if __name__ == "__main__":
    unittest.main()


