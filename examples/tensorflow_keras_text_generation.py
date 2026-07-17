"""TensorFlow backend ve KerasHub ile komut satırından metin üretimi."""

from __future__ import annotations

import argparse
import os
import time

# Backend, Keras veya KerasHub import edilmeden önce seçilmelidir.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402
import keras_hub  # noqa: E402
import tensorflow as tf  # noqa: E402


DEFAULT_MODEL = "gemma3_instruct_270m"
DEFAULT_PROMPT = "Büyük dil modellerinde attention mekanizmasını kısa biçimde açıkla."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KerasHub CausalLM preset'iyle metin üretir."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="KerasHub preset kimliği")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Üretim promptu")
    parser.add_argument(
        "--max-length",
        type=int,
        default=192,
        help="Prompt dahil en yüksek dizi uzunluğu",
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Top-p sampling yerine greedy üretim kullan",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.max_length < 2:
        raise ValueError("--max-length en az 2 olmalıdır.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p 0 ile 1 arasında olmalıdır.")
    if args.top_k < 1:
        raise ValueError("--top-k en az 1 olmalıdır.")

    print(f"Keras backend: {keras.backend.backend()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU'lar: {tf.config.list_physical_devices('GPU')}")
    print(f"Model: {args.model}")

    model = keras_hub.models.CausalLM.from_preset(args.model)

    if args.greedy:
        model.compile(sampler="greedy")
    else:
        sampler = keras_hub.samplers.TopPSampler(
            p=args.top_p,
            k=args.top_k,
            seed=args.seed,
        )
        model.compile(sampler=sampler)

    # İlk çağrı derleme ve ısınma maliyeti içerebilir.
    start = time.perf_counter()
    answer = model.generate(
        args.prompt,
        max_length=args.max_length,
        strip_prompt=True,
    )
    elapsed = time.perf_counter() - start

    print("\n--- Cevap ---")
    print(answer)
    print("\n--- Ölçüm ---")
    print(f"İlk çağrı süresi: {elapsed:.2f} saniye")
    print("Karşılaştırma için aynı promptu birkaç kez çalıştırıp ısınmış süreyi ölçün.")


if __name__ == "__main__":
    main()


