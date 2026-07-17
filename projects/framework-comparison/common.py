"""PyTorch ve TensorFlow karşılaştırması için ortak veri işlemleri."""

from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


TOKEN_PATTERN = re.compile(r"[0-9a-zçğıöşü]+", flags=re.IGNORECASE)


@dataclass(frozen=True)
class SentimentExample:
    text: str
    label: int


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


def tokenize(text: str) -> list[str]:
    """Türkçe karakterleri koruyan basit ve deterministik tokenizer."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("i\u0307", "i")
    return TOKEN_PATTERN.findall(normalized)


def load_examples(path: Path) -> list[SentimentExample]:
    """text,label sütunlarına sahip CSV dosyasını doğrulayarak yükle."""
    examples: list[SentimentExample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "label"]:
            raise ValueError("CSV başlığı tam olarak 'text,label' olmalıdır.")

        for line_number, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            if not text:
                raise ValueError(f"{line_number}. satırda metin boş.")

            try:
                label = int(row.get("label", ""))
            except ValueError as exc:
                raise ValueError(f"{line_number}. satırda etiket sayı değil.") from exc

            if label not in {0, 1}:
                raise ValueError(f"{line_number}. satırda etiket 0 veya 1 olmalıdır.")
            examples.append(SentimentExample(text=text, label=label))

    if len(examples) < 4:
        raise ValueError("En az dört örnek gereklidir.")
    if {example.label for example in examples} != {0, 1}:
        raise ValueError("Veri kümesi hem 0 hem 1 etiketini içermelidir.")
    return examples


def stratified_split(
    examples: Sequence[SentimentExample],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[SentimentExample], list[SentimentExample]]:
    """Her sınıfı koruyarak deterministik train/test bölmesi oluştur."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio 0 ile 1 arasında olmalıdır.")

    randomizer = random.Random(seed)
    by_label: dict[int, list[SentimentExample]] = {0: [], 1: []}
    for example in examples:
        by_label[example.label].append(example)

    train: list[SentimentExample] = []
    test: list[SentimentExample] = []
    for label, group in by_label.items():
        if len(group) < 2:
            raise ValueError(f"{label} etiketi için en az iki örnek gereklidir.")
        shuffled = list(group)
        randomizer.shuffle(shuffled)
        test_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_ratio)))
        test.extend(shuffled[:test_count])
        train.extend(shuffled[test_count:])

    randomizer.shuffle(train)
    randomizer.shuffle(test)
    return train, test


def build_vocabulary(
    examples: Iterable[SentimentExample],
    max_features: int = 2_000,
    min_frequency: int = 1,
) -> dict[str, int]:
    """Yalnızca eğitim örneklerinden deterministik kelime sözlüğü oluştur."""
    if max_features < 1:
        raise ValueError("max_features en az 1 olmalıdır.")
    if min_frequency < 1:
        raise ValueError("min_frequency en az 1 olmalıdır.")

    counts: Counter[str] = Counter()
    for example in examples:
        counts.update(tokenize(example.text))

    selected = [
        token
        for token, frequency in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency >= min_frequency
    ][:max_features]
    if not selected:
        raise ValueError("Sözlük boş kaldı; veri ve min_frequency değerini kontrol edin.")
    return {token: index for index, token in enumerate(selected)}


def vectorize_text(text: str, vocabulary: dict[str, int]) -> list[float]:
    """Metni normalize edilmiş kelime sıklığı vektörüne dönüştür."""
    vector = [0.0] * len(vocabulary)
    known_tokens = [token for token in tokenize(text) if token in vocabulary]
    if not known_tokens:
        return vector

    normalizer = float(len(known_tokens))
    for token in known_tokens:
        vector[vocabulary[token]] += 1.0 / normalizer
    return vector


def vectorize_examples(
    examples: Sequence[SentimentExample], vocabulary: dict[str, int]
) -> tuple[list[list[float]], list[int]]:
    features = [vectorize_text(example.text, vocabulary) for example in examples]
    labels = [example.label for example in examples]
    return features, labels


def binary_classification_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> BinaryMetrics:
    """Pozitif sınıf 1 kabul edilerek temel sınıflandırma metriklerini hesapla."""
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Etiket ve tahminler aynı ve sıfırdan büyük uzunlukta olmalıdır.")
    if set(labels) - {0, 1} or set(predictions) - {0, 1}:
        raise ValueError("Etiket ve tahminler yalnızca 0 veya 1 içermelidir.")

    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def save_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def metrics_to_dict(metrics: BinaryMetrics) -> dict[str, object]:
    return asdict(metrics)


