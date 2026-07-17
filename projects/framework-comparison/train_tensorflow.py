"""Ortak Türkçe veri kümesinde küçük bir TensorFlow/Keras sınıflandırıcı eğit."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from common import (  # noqa: E402
    binary_classification_metrics,
    build_vocabulary,
    load_examples,
    metrics_to_dict,
    save_result,
    stratified_split,
    vectorize_examples,
)


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TensorFlow/Keras duygu analizi karşılaştırması"
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT_DIR / "data" / "tr_sentiment.csv"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epoch ve batch en az 1; learning rate sıfırdan büyük olmalıdır.")

    keras.utils.set_random_seed(args.seed)

    examples = load_examples(args.data)
    train_examples, test_examples = stratified_split(
        examples, test_ratio=args.test_ratio, seed=args.seed
    )
    vocabulary = build_vocabulary(train_examples)
    x_train, y_train = vectorize_examples(train_examples, vocabulary)
    x_test, y_test = vectorize_examples(test_examples, vocabulary)

    train_features = tf.convert_to_tensor(x_train, dtype=tf.float32)
    train_labels = tf.convert_to_tensor(y_train, dtype=tf.int32)
    test_features = tf.convert_to_tensor(x_test, dtype=tf.float32)

    model = keras.Sequential(
        [
            keras.Input(shape=(len(vocabulary),)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(2),
        ],
        name="turkish_sentiment_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    start = time.perf_counter()
    history = model.fit(
        train_features,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        shuffle=True,
    )
    training_seconds = time.perf_counter() - start

    logits = model(test_features, training=False)
    predictions = tf.argmax(logits, axis=1).numpy().tolist()
    metrics = binary_classification_metrics(y_test, predictions)

    result: dict[str, object] = {
        "framework": "TensorFlow/Keras",
        "devices": [device.name for device in tf.config.list_logical_devices()],
        "seed": args.seed,
        "train_size": len(train_examples),
        "test_size": len(test_examples),
        "vocabulary_size": len(vocabulary),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_epoch_loss": float(history.history["loss"][-1]),
        "training_seconds": training_seconds,
        "metrics": metrics_to_dict(metrics),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_result(args.output_dir / "tensorflow_metrics.json", result)
    model.save(args.output_dir / "tensorflow_sentiment.keras")
    save_result(args.output_dir / "tensorflow_vocabulary.json", vocabulary)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


