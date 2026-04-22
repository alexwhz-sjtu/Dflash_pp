#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DFlash++ test-in-loop：冒烟 → 无树快速网格（截断×迭代）→ 最优线稿 vs 草稿树 A/B 吞吐比。

推荐：`source .venv/bin/activate`；或 conda `eagle`。可 export CUDA_VISIBLE_DEVICES=...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_PY = REPO / "eval" / "eval.py"
DEFAULT_QUESTION = (
    "/share/wanghanzhen/SpeculativeDecoding/NIPS26/dataset/mtbench101/question.jsonl"
)
OUT = REPO / "eval" / "_til_out"
TRUNCK = REPO / "eval" / "trunck"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or REPO, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-file", type=str, default=DEFAULT_QUESTION)
    ap.add_argument("--begin", type=int, default=0)
    ap.add_argument("--end", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--grid-max-new-tokens", type=int, default=512)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--skip-tree-ab", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"repo": str(REPO), "phases": []}

    base = [
        py,
        str(EVAL_PY),
        "--model-pair",
        "dflash-pp-trained",
        "--question-file",
        args.question_file,
        "--begin",
        str(args.begin),
        "--end",
        str(args.end),
        "--max-length",
        str(args.max_length),
        "--truncation-policy",
        "min_prob",
        "--max-block-inner-iters",
        "3",
        "--trunc-tau",
        "0.88",
    ]

    if not args.skip_smoke:
        smoke_dir = OUT / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        summ_path = OUT / "smoke_summary.json"
        _run(
            base
            + [
                "--output-dir",
                str(smoke_dir),
                "--emit-run-summary-json",
                str(summ_path),
            ]
        )
        report["smoke_summary"] = json.loads(summ_path.read_text(encoding="utf-8"))
        report["phases"].append({"name": "smoke", "summary_path": str(summ_path)})

    if not args.skip_grid:
        TRUNCK.mkdir(parents=True, exist_ok=True)
        grid_dir = TRUNCK / "grid_fast"
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_json = grid_dir / "grid_truncation_results.json"
        _run(
            [
                py,
                str(EVAL_PY),
                "--model-pair",
                "dflash-pp-trained",
                "--question-file",
                args.question_file,
                "--begin",
                str(args.begin),
                "--end",
                str(args.end),
                "--grid-truncation",
                "--grid-preset",
                "fast",
                "--grid-sort-by",
                "composite",
                "--grid-max-new-tokens",
                str(args.grid_max_new_tokens),
                "--output-dir",
                str(grid_dir),
            ]
        )
        grid_rows = json.loads(grid_json.read_text(encoding="utf-8"))
        report["grid_fast_top5"] = grid_rows[:5]
        report["phases"].append(
            {"name": "grid_fast_no_tree", "path": str(grid_json), "n": len(grid_rows)}
        )
        best_cfg = dict(grid_rows[0]["config"])
        best_line_path = TRUNCK / "best_line_spec.json"
        best_line_path.write_text(
            json.dumps(best_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["best_line_spec"] = best_cfg
    else:
        best_line_path = TRUNCK / "best_line_spec.json"
        if not best_line_path.is_file():
            print("缺少 best_line_spec.json，请先跑网格或去掉 --skip-grid", file=sys.stderr)
            sys.exit(1)
        best_cfg = json.loads(best_line_path.read_text(encoding="utf-8"))

    if not args.skip_tree_ab:
        line_dir = OUT / "ab_line"
        tree_dir = OUT / "ab_tree"
        line_dir.mkdir(parents=True, exist_ok=True)
        tree_dir.mkdir(parents=True, exist_ok=True)
        line_sum = OUT / "ab_line_summary.json"
        tree_sum = OUT / "ab_tree_summary.json"

        _run(
            [
                py,
                str(EVAL_PY),
                "--model-pair",
                "dflash-pp-trained",
                "--question-file",
                args.question_file,
                "--begin",
                str(args.begin),
                "--end",
                str(args.end),
                "--max-length",
                str(args.max_length),
                "--output-dir",
                str(line_dir),
                "--spec-json",
                str(best_line_path),
                "--emit-run-summary-json",
                str(line_sum),
            ]
        )
        tree_cfg = dict(best_cfg)
        tree_cfg["use_draft_tree"] = True
        tree_spec_path = TRUNCK / "best_tree_spec.json"
        tree_spec_path.write_text(
            json.dumps(tree_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _run(
            [
                py,
                str(EVAL_PY),
                "--model-pair",
                "dflash-pp-trained",
                "--question-file",
                args.question_file,
                "--begin",
                str(args.begin),
                "--end",
                str(args.end),
                "--max-length",
                str(args.max_length),
                "--output-dir",
                str(tree_dir),
                "--spec-json",
                str(tree_spec_path),
                "--emit-run-summary-json",
                str(tree_sum),
            ]
        )
        sl = json.loads(line_sum.read_text(encoding="utf-8"))
        st = json.loads(tree_sum.read_text(encoding="utf-8"))
        tl, tt = sl["overall_throughput"], st["overall_throughput"]
        report["ab_line_summary"] = sl
        report["ab_tree_summary"] = st
        report["draft_tree_speedup_vs_best_line"] = (tt / tl) if tl > 0 else None
        report["phases"].append(
            {
                "name": "tree_ablation",
                "line_throughput": tl,
                "tree_throughput": tt,
                "speedup": report["draft_tree_speedup_vs_best_line"],
            }
        )

    rep_path = OUT / "test_in_loop_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {rep_path}", flush=True)
    if "draft_tree_speedup_vs_best_line" in report:
        print(
            f"草稿树相对最优线稿吞吐加速比: {report['draft_tree_speedup_vs_best_line']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
