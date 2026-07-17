"""PyTorch ve Transformers ile komut satırından metin üretimi."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_PROMPT = "Büyük dil modellerinde tokenizasyonu kısa ve anlaşılır biçimde açıkla."


def select_device() -> torch.device:
    """Kullanılabilir en uygun PyTorch cihazını seç."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Küçük bir instruction modeliyle metin üretir."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model kimliği")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Kullanıcı mesajı")
    parser.add_argument(
        "--max-new-tokens", type=int, default=160, help="En fazla yeni token"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Sampling yerine deterministik greedy üretim kullan",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens en az 1 olmalıdır.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p 0 ile 1 arasında olmalıdır.")
    if not args.greedy and args.temperature <= 0:
        raise ValueError("Sampling için --temperature sıfırdan büyük olmalıdır.")

    torch.manual_seed(args.seed)
    device = select_device()

    print(f"Model: {args.model}")
    print(f"Cihaz: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_args: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if not args.greedy:
        generation_args.update(
            temperature=args.temperature,
            top_p=args.top_p,
        )

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_args)
    elapsed = time.perf_counter() - start

    prompt_length = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[0, prompt_length:]
    answer = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    token_count = int(new_token_ids.shape[0])
    tokens_per_second = token_count / elapsed if elapsed else 0.0

    print("\n--- Cevap ---")
    print(answer)
    print("\n--- Ölçüm ---")
    print(f"Yeni token: {token_count}")
    print(f"Süre: {elapsed:.2f} saniye")
    print(f"Hız: {tokens_per_second:.2f} token/saniye")


if __name__ == "__main__":
    main()


