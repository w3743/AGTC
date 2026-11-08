"""Predict digits online with the all-source input-memory RTRL model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
import torch
from input_memory_rtrl import InputMemoryRTRL


def process(model: InputMemoryRTRL, digit: int) -> None:
    probabilities = model.predict().tolist()
    prediction = max(range(10), key=lambda i: (probabilities[i], -i))
    print(f"预测={prediction} 输入={digit} {'命中' if prediction == digit else '未命中'} "
          f"概率=[{', '.join(f'{p:.4f}' for p in probabilities)}]", flush=True)
    model.learn_target(digit)
    model.advance(digit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", help="输入一串0–9；省略时逐位交互输入")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.digits is not None and (not args.digits or any(c not in "0123456789" for c in args.digits)):
        parser.error("--digits 必须是非空的0–9数字串")
    torch.set_num_threads(1)
    model = InputMemoryRTRL(seed=args.seed, scope="all", mode="learned")
    if args.digits is not None:
        for char in args.digits:
            process(model, int(char))
        return
    while True:
        try:
            value = input("输入一位0–9（q退出）：").strip()
        except EOFError:
            break
        if value.lower() == "q":
            break
        if len(value) != 1 or value not in "0123456789":
            print("请输入一位0–9，或输入q退出。")
            continue
        process(model, int(value))


if __name__ == "__main__":
    main()
