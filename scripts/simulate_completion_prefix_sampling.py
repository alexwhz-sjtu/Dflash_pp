#!/usr/bin/env python3
# coding=utf-8
"""
模拟 OnlineDFlashPPModel 中 L_con 前缀长度 p 的采样分布。

与训练一致: logits_p = -w * (p - 1 - b)^2,  P(p) = softmax(logits), p ∈ {1, ..., B-1}。
（线性 -w*(p-1-b) 中 +w*b 对 softmax 为常数，b 不会改变分布；故训练侧已改为平方形式。）

用法示例:
  python scripts/simulate_completion_prefix_sampling.py --block-size 16 --weight 1 --bias 0
  python scripts/simulate_completion_prefix_sampling.py -B 16 --multi "1,0;0,0;2,0.5" --mc 100000
"""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Sequence, Tuple


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def theoretical_probs(
    block_size: int, weight: float, bias: float
) -> Tuple[List[int], List[float], List[float]]:
    """返回 (p 列表, P(p), 累积概率)."""
    if block_size < 2:
        raise ValueError("block_size 必须 >= 2")
    ps = list(range(1, block_size))
    logits = [-weight * (p - 1.0 - bias) ** 2 for p in ps]
    probs = softmax(logits)
    cum = []
    acc = 0.0
    for p in probs:
        acc += p
        cum.append(acc)
    return ps, probs, cum


def monte_carlo_counts(
    block_size: int,
    weight: float,
    bias: float,
    n_samples: int,
    seed: int | None = None,
) -> List[int]:
    """按与训练相同的分布多项式采样，返回每个 p 的出现次数。"""
    ps, probs, _ = theoretical_probs(block_size, weight, bias)
    rng = random.Random(seed)
    counts = [0] * len(ps)
    for _ in range(n_samples):
        r = rng.random()
        acc = 0.0
        for i, pr in enumerate(probs):
            acc += pr
            if r <= acc:
                counts[i] += 1
                break
    return counts


def parse_multi(s: str) -> List[Tuple[float, float]]:
    """格式: "w1,b1;w2,b2" """
    out: List[Tuple[float, float]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(",")
        out.append((float(a.strip()), float(b.strip())))
    return out


def print_table(
    label: str,
    ps: List[int],
    probs: List[float],
    cum: List[float],
    counts: List[int] | None = None,
    n_mc: int = 0,
) -> None:
    print(f"\n=== {label} ===")
    col = "  p    P(p)     累积"
    if counts is not None and n_mc > 0:
        col += f"    MC频率(n={n_mc})"
    print(col)
    print("-" * (56 if counts is None else 72))
    for i, p in enumerate(ps):
        line = f"  {p:2d}   {probs[i]:.6f}   {cum[i]:.6f}"
        if counts is not None and n_mc > 0:
            line += f"   {counts[i] / n_mc:.6f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计/模拟 L_con 前缀 p 的采样概率 (与 dflash_pp._sample_prefix_lengths 一致)"
    )
    parser.add_argument(
        "-B",
        "--block-size",
        type=int,
        default=16,
        help="块大小 B，合法 p 为 1..B-1",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        default=0.1,
        help="completion_prefix_sample_weight",
    )
    parser.add_argument(
        "-b",
        "--bias",
        type=float,
        default=1.0,
        help="completion_prefix_sample_bias",
    )
    parser.add_argument(
        "--multi",
        type=str,
        default=None,
        help='多组 (w,b)，分号分隔，如 "1,0;0,0;2,1.5"',
    )
    parser.add_argument(
        "--mc",
        type=int,
        default=0,
        help="蒙特卡洛采样次数；0 表示不做模拟，只打理论概率",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="--mc>0 时的随机种子",
    )
    args = parser.parse_args()

    scenarios: List[Tuple[str, float, float]] = []
    if args.multi:
        for i, (w, b) in enumerate(parse_multi(args.multi)):
            scenarios.append((f"w={w}, b={b}", w, b))
    else:
        scenarios.append((f"w={args.weight}, b={args.bias}", args.weight, args.bias))

    print(
        f"块大小 B={args.block_size}，p ∈ {{1,…,{args.block_size - 1}}}，"
        f"logits_p = -w·(p-1-b)²，P = softmax(logits)"
    )

    for si, (label, w, b) in enumerate(scenarios):
        ps, probs, cum = theoretical_probs(args.block_size, w, b)
        counts: List[int] | None = None
        if args.mc > 0:
            counts = monte_carlo_counts(
                args.block_size,
                w,
                b,
                args.mc,
                seed=args.seed + si * 1_000_003,
            )
        print_table(label, ps, probs, cum, counts=counts, n_mc=args.mc)


if __name__ == "__main__":
    main()
