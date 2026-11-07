"""Fixed 600-digit next-symbol tasks for the decimal RNN benchmark.

All task sequences are deterministic and shared across model seeds.  The IID
task uses a private ``random.Random`` instance, so importing or calling this
module never changes Python's process-wide random state.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import random
from typing import Any


SEQUENCE_LENGTH = 600
SEEDS = [2, 20261027, 20261028]
_IID_SEED = 891732849
_IID_SHA256 = "75b38d99e6f3340ee8dd6c5fccfbe1c67f7252fe22db89b73e22cc582a4b2715"
_IID_HISTOGRAM = {0: 48, 1: 56, 2: 57, 3: 51, 4: 69, 5: 65, 6: 69, 7: 68, 8: 62, 9: 55}


TASKS: dict[str, dict[str, Any]] = {
    "cycle": {
        "name_zh": "十进制循环",
        "description": "重复 0 到 9 的升序循环，每个数字后继唯一。",
        "example": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        "stationary": True,
        "random_baseline": None,
    },
    "triples": {
        "name_zh": "三连数字块",
        "description": "每个数字连续重复三次，按 0 到 9 循环。",
        "example": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "stationary": True,
        "random_baseline": None,
    },
    "zigzag": {
        "name_zh": "往返锯齿",
        "description": "重复 012345678987654321，共 18 位；0 和 9 各出现一次。",
        "example": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "stationary": True,
        "random_baseline": None,
    },
    "fibonacci_mod10": {
        "name_zh": "斐波那契模十",
        "description": "从 1、2 开始，后续数字等于前两位之和模 10。",
        "example": [1, 2, 3, 5, 8, 3, 1, 4, 5, 9, 4, 3],
        "stationary": True,
        "random_baseline": None,
    },
    "switching": {
        "name_zh": "三阶段切换",
        "description": "无状态重置地拼接三个各 200 位的阶段：正向步长 1、逆向步长 1、正向步长 3。",
        "example": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        "stationary": False,
        "random_baseline": None,
        "phase_bounds": [[1, 200], [201, 400], [401, 600]],
    },
    "iid_random": {
        "name_zh": "固定均匀随机",
        "description": "由独立均匀的十进制抽样生成并固定为同一条 600 位序列。",
        "example": [6, 9, 8, 7, 6, 0, 6, 4, 6, 1, 5, 4],
        "stationary": True,
        "random_baseline": 0.1,
        "generator_seed": _IID_SEED,
        "sha256": _IID_SHA256,
        "histogram": {str(k): v for k, v in _IID_HISTOGRAM.items()},
    },
}

_PATTERNS = {
    "cycle": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "triples": [digit for digit in range(10) for _ in range(3)],
    "zigzag": list(range(10)) + list(range(8, 0, -1)),
    "step3": [0, 3, 6, 9, 2, 5, 8, 1, 4, 7],
}


def _iid_sequence() -> list[int]:
    rng = random.Random(_IID_SEED)
    return [rng.randrange(10) for _ in range(SEQUENCE_LENGTH)]


def make_sequence(task: str, length: int = 600) -> list[int]:
    """Return the fixed sequence for ``task``.

    This benchmark intentionally accepts only 600 digits, preserving the
    switching task's exact phase boundaries and equal data budgets.
    """
    if length != SEQUENCE_LENGTH:
        raise ValueError(f"length must be exactly {SEQUENCE_LENGTH} for benchmark fairness")
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")

    if task == "fibonacci_mod10":
        seq = [1, 2]
        for _ in range(length - 2):
            seq.append((seq[-1] + seq[-2]) % 10)
        return seq
    if task == "switching":
        forward = _PATTERNS["cycle"]
        reverse = list(reversed(forward))
        stepped = _PATTERNS["step3"]
        return (
            [forward[i % 10] for i in range(200)]
            + [reverse[i % 10] for i in range(200)]
            + [stepped[i % 10] for i in range(200)]
        )
    if task == "iid_random":
        return _iid_sequence()
    pattern = _PATTERNS[task]
    return [pattern[i % len(pattern)] for i in range(length)]


def validate() -> dict[str, Any]:
    """Run task invariants and return a JSON-serializable audit summary."""
    sequences = {name: make_sequence(name) for name in TASKS}
    for name, seq in sequences.items():
        assert len(seq) == SEQUENCE_LENGTH, (name, len(seq))
        assert all(type(value) is int and 0 <= value <= 9 for value in seq), name
        assert set(seq) == set(range(10)), f"{name} must cover all decimal digits"

    for name, pattern in _PATTERNS.items():
        if name == "step3":
            continue
        seq = sequences[name]
        assert seq == [pattern[i % len(pattern)] for i in range(SEQUENCE_LENGTH)], name

    fib = sequences["fibonacci_mod10"]
    assert fib[:2] == [1, 2]
    assert all(fib[i] == (fib[i - 1] + fib[i - 2]) % 10 for i in range(2, len(fib)))

    zigzag = sequences["zigzag"]
    successors: dict[int, set[int]] = {}
    for current, following in zip(zigzag, zigzag[1:] + zigzag[:1]):
        successors.setdefault(current, set()).add(following)
    assert any(len(next_digits) > 1 for next_digits in successors.values()), successors
    assert Counter(zigzag)[0] == 34 and Counter(zigzag)[9] == 33

    switching = sequences["switching"]
    assert switching[:200] == [i % 10 for i in range(200)]
    assert switching[200:400] == [9 - (i % 10) for i in range(200)]
    assert switching[400:] == [_PATTERNS["step3"][i % 10] for i in range(200)]
    assert switching[199] == 9 and switching[200] == 9  # no reset between phases
    assert switching[399] == 0 and switching[400] == 0

    iid = sequences["iid_random"]
    histogram = dict(sorted(Counter(iid).items()))
    digest = sha256(bytes(iid)).hexdigest()
    assert histogram == _IID_HISTOGRAM, histogram
    assert digest == _IID_SHA256, digest

    return {
        "valid": True,
        "length": SEQUENCE_LENGTH,
        "tasks": list(TASKS),
        "digits_per_task": {name: sorted(set(seq)) for name, seq in sequences.items()},
        "switching_phase_bounds": TASKS["switching"]["phase_bounds"],
        "iid_random": {
            "sha256": digest,
            "histogram": {str(k): v for k, v in histogram.items()},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate(), ensure_ascii=False, indent=2))
