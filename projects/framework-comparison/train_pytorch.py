"""Ortak Türkçe veri kümesinde küçük bir PyTorch sınıflandırıcı eğit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
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
    parser = argparse.ArgumentParser(description="PyTorch duygu analizi karşılaştırması")
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


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epoch ve batch en az 1; learning rate sıfırdan büyük olmalıdır.")

    torch.manual_seed(args.seed)
    device = select_device()

    examples = load_examples(args.data)
    train_examples, test_examples = stratified_split(
        examples, test_ratio=args.test_ratio, seed=args.seed
    )
    vocabulary = build_vocabulary(train_examples)
    x_train, y_train = vectorize_examples(train_examples, vocabulary)
    x_test, y_test = vectorize_examples(test_examples, vocabulary)

    train_features = torch.tensor(x_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_features = torch.tensor(x_test, dtype=torch.float32, device=device)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = nn.Sequential(
        nn.Linear(len(vocabulary), 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 2),
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start = time.perf_counter()
    model.train()
    final_loss = 0.0
    for _ in range(args.epochs):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    training_seconds = time.perf_counter() - start

    model.eval()
    with torch.inference_mode():
        predictions = model(test_features).argmax(dim=1).cpu().tolist()
    metrics = binary_classification_metrics(y_test, predictions)

    result: dict[str, object] = {
        "framework": "PyTorch",
        "device": str(device),
        "seed": args.seed,
        "train_size": len(train_examples),
        "test_size": len(test_examples),
        "vocabulary_size": len(vocabulary),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_batch_loss": final_loss,
        "training_seconds": training_seconds,
        "metrics": metrics_to_dict(metrics),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_result(args.output_dir / "pytorch_metrics.json", result)
    torch.save(
        {"state_dict": model.state_dict(), "vocabulary": vocabulary},
        args.output_dir / "pytorch_sentiment.pt",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


