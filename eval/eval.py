"""DFlash++ 测评：多轮块内迭代、截断策略、可选草稿树与网格搜索（请在 conda eagle 环境中运行）。"""
import os
import sys
import importlib.util

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRUNCK_DIR = os.path.join(_REPO, "eval", "trunck")
_DEFAULT_TRUNC_PROFILE_DRAFT_TOPK = os.path.join(_TRUNCK_DIR, "draft_topk.jsonl")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
import torch
import json
import argparse
import numpy as np
from itertools import product
from typing import Any, Dict, List, Optional

# 避免 import specforge 包（会拉 sglang/eagle 依赖）；仅加载 dflash 草稿实现
_DFLASH_PATH = os.path.join(_REPO, "specforge", "modeling", "draft", "dflash.py")
_dflash_spec = importlib.util.spec_from_file_location(
    "_eval_dflash_draft_only", _DFLASH_PATH
)
assert _dflash_spec and _dflash_spec.loader
_dflash_mod = importlib.util.module_from_spec(_dflash_spec)
_dflash_spec.loader.exec_module(_dflash_mod)
DFlashDraftModel = _dflash_mod.DFlashDraftModel

_DEFAULT_DRAFT_PP = (
    "/share/wanghanzhen/SpeculativeDecoding/NIPS26/DFlash++/cache/models/"
    "DFlash_pp_sample_400000_think_on_qwen3_8b_maxlen4096_epochs16_1/epoch_3_step_140000"
)


def register_local_dflash_model():
    try:
        AutoModel.register(Qwen3Config, DFlashDraftModel, exist_ok=True)
    except TypeError:
        AutoModel.register(Qwen3Config, DFlashDraftModel)


def load_draft_qwen3_config(draft_model_path: str) -> Qwen3Config:
    """从 config.json 加载并修补 layer_types，避免 AutoConfig 在校验阶段失败。"""
    cfg_path = os.path.join(draft_model_path, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    n = int(cfg_dict.get("num_hidden_layers", 0) or 0)
    lt = cfg_dict.get("layer_types")
    if n > 0 and isinstance(lt, list) and len(lt) != n:
        cfg_dict["layer_types"] = ["full_attention"] * n
    mw = cfg_dict.get("max_window_layers")
    if mw is not None and n > 0 and int(mw) > n:
        cfg_dict["max_window_layers"] = n
    return Qwen3Config(**cfg_dict)


def load_mtbench101_questions(question_file, begin=None, end=None):
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                turns = []
                for hist in data["history"]:
                    turns.append(hist["user"])
                questions.append(
                    {
                        "question_id": data["id"],
                        "category": data.get("task", "mtbench101"),
                        "turns": turns,
                        "history": data["history"],
                    }
                )
    if begin is not None and end is not None:
        questions = questions[begin:end]
    return questions


def multi_turn_dialogue(
    draft_model,
    target_model,
    tokenizer,
    turns,
    max_new_tokens=4096,
    temperature=0.0,
    log_file=None,
    thinking=False,
    debug_dir=None,
    draft_topk_file=None,
    draft_topk=2,
    spec_generate_kw: Optional[Dict[str, Any]] = None,
):
    conversation_history = []
    responses = []
    turn_stats = []
    base_kw: Dict[str, Any] = {
        "max_block_inner_iters": 1,
        "truncation_policy": "full",
        "trunc_tau": 0.85,
        "trunc_risk_eps": 0.35,
        "trunc_score_beta4": 0.15,
        "inner_fallback": "one",
        "use_draft_tree": False,
        "draft_tree_branches": 3,
        "debug_dir": debug_dir,
        "tokenizer": tokenizer,
        "draft_topk_file": draft_topk_file,
        "draft_topk": draft_topk,
    }
    if spec_generate_kw:
        base_kw.update(spec_generate_kw)

    for turn_idx, user_input in enumerate(turns):
        if log_file:
            log_file.write(f"Turn {turn_idx + 1}: {user_input[:50]}...\n")
            log_file.flush()

        conversation_history.append(
            {
                "role": "user",
                "content": "Answer the following question as detailed as possible: "
                + user_input,
            }
        )
        text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(
            next(draft_model.parameters()).device
        )

        output_ids = draft_model.spec_generate(
            target=target_model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=temperature,
            **base_kw,
        )

        output = tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        conversation_history.append({"role": "assistant", "content": output})
        responses.append(output)

        response_token_ids = tokenizer([output]).input_ids[0]
        response_length = len(response_token_ids)
        get_stats_fn = getattr(draft_model, "get_last_decode_stats", None)
        stats = get_stats_fn() if callable(get_stats_fn) else None
        stats = stats or {
            "accept_lengths": [],
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
        }

        stats["response_length"] = response_length
        accept_lengths = stats.get("accept_lengths", [])
        stats["mean_accept_length"] = (
            float(np.mean(accept_lengths)) if accept_lengths else 0.0
        )
        total_time = stats.get("target_total_time", 0.0) + stats.get(
            "draft_total_time", 0.0
        )
        stats["throughput"] = (
            response_length / total_time if total_time > 0 and response_length > 0 else 0.0
        )
        turn_stats.append(stats)

        if log_file:
            log_file.write(f"Assistant: {output[:100]}...\n")
            log_file.write(f"  Response Length: {response_length} tokens\n")
            log_file.write(f"  Accept Lengths: {stats['accept_lengths']}\n")
            log_file.write(f"  Mean Accept Length: {stats['mean_accept_length']:.4f}\n")
            log_file.write(
                f"  mean_second_plus_kt (块内第2轮及以后多接收的 token): "
                f"{stats.get('mean_second_plus_kt', 0.0):.4f}\n"
            )
            log_file.write(
                f"  trunc_commit≤accept 后验命中率: "
                f"{stats.get('trunc_commit_le_accept_rate', 0.0):.4f}\n"
            )
            log_file.write(
                f"  inner_kt_rounds (每验证步各轮 k): {stats.get('inner_kt_rounds', [])}\n"
            )
            if stats.get("record_posthoc_suffix_refine"):
                log_file.write(
                    f"  posthoc_suffix_refine: mean_gain_corrected="
                    f"{stats.get('mean_posthoc_suffix_accept_gain_corrected', stats.get('mean_posthoc_suffix_accept_gain', 0)):.4f} | "
                    f"executed={stats.get('n_posthoc_suffix_refine_executed', stats.get('n_posthoc_suffix_events', 0))} | "
                    f"skipped_anchor_only={stats.get('n_posthoc_suffix_skipped_anchor_only', 0)} | "
                    f"skipped_acc1_out_of_range={stats.get('n_posthoc_suffix_skipped_acc1_out_of_range', 0)} | "
                    f"acc1_range=[{stats.get('posthoc_acc1_min', 2)},{stats.get('posthoc_acc1_max', 14)}] | "
                    f"target_extra_s={stats.get('target_posthoc_extra_time', 0):.4f}\n"
                )
            log_file.write(
                f"  Target Time: {stats['target_total_time']:.4f}s | "
                f"Draft Time: {stats['draft_total_time']:.4f}s\n"
            )
            log_file.write(f"  Throughput: {stats['throughput']:.2f} tokens/sec\n\n")
            log_file.flush()

    return responses, turn_stats


def summarize_question_stats(turn_stats, responses=None):
    all_accept_lengths = []
    response_lengths = []
    mean_accept_lengths = []
    throughputs = []
    target_total = 0.0
    draft_total = 0.0
    total_steps = 0
    total_response_length = 0
    second_plus_all = []
    calib_rates = []
    mean_inner_iters_list = []
    all_inner_kt_rounds = []
    posthoc_gains_flat: List[float] = []
    posthoc_pairs_flat: List[List[float]] = []
    posthoc_skipped_anchor_total = 0
    posthoc_skipped_acc1_oor_total = 0
    posthoc_executed_total = 0
    target_posthoc_extra = 0.0

    for one_turn in turn_stats:
        all_accept_lengths.extend(one_turn.get("accept_lengths", []))
        response_lengths.append(one_turn.get("response_length", 0))
        mean_accept_lengths.append(one_turn.get("mean_accept_length", 0.0))
        throughputs.append(one_turn.get("throughput", 0.0))
        target_total += one_turn.get("target_total_time", 0.0)
        draft_total += one_turn.get("draft_total_time", 0.0)
        total_steps += one_turn.get("steps", 0)
        total_response_length += one_turn.get("response_length", 0)
        second_plus_all.extend(one_turn.get("second_plus_kt_values", []))
        if "trunc_commit_le_accept_rate" in one_turn:
            calib_rates.append(float(one_turn["trunc_commit_le_accept_rate"]))
        mean_inner_iters_list.append(float(one_turn.get("mean_inner_iters", 0.0)))
        for rnd in one_turn.get("inner_kt_rounds", []):
            all_inner_kt_rounds.append(rnd)
        for g in one_turn.get("posthoc_suffix_refine_gains", []):
            if g is not None:
                posthoc_gains_flat.append(float(g))
        for pair in one_turn.get("posthoc_suffix_refine_pairs", []):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                posthoc_pairs_flat.append([float(pair[0]), float(pair[1])])
        posthoc_skipped_anchor_total += int(
            one_turn.get("n_posthoc_suffix_skipped_anchor_only", 0)
        )
        posthoc_skipped_acc1_oor_total += int(
            one_turn.get("n_posthoc_suffix_skipped_acc1_out_of_range", 0)
        )
        posthoc_executed_total += int(
            one_turn.get("n_posthoc_suffix_refine_executed", 0)
        )
        target_posthoc_extra += float(one_turn.get("target_posthoc_extra_time", 0.0))

    mean_accept = float(np.mean(all_accept_lengths)) if all_accept_lengths else 0.0
    mean_response_length = float(np.mean(response_lengths)) if response_lengths else 0.0
    mean_per_turn_accept = (
        float(np.mean(mean_accept_lengths)) if mean_accept_lengths else 0.0
    )
    mean_throughput = float(np.mean(throughputs)) if throughputs else 0.0
    total_time = target_total + draft_total
    overall_throughput = (
        total_response_length / total_time if total_time > 0 else 0.0
    )
    mean_second_plus_kt = (
        float(np.mean(second_plus_all)) if second_plus_all else 0.0
    )
    mean_trunc_calib = float(np.mean(calib_rates)) if calib_rates else 0.0
    mean_inner_iters = (
        float(np.mean(mean_inner_iters_list)) if mean_inner_iters_list else 0.0
    )

    kt2_only = []
    for rnd in all_inner_kt_rounds:
        if len(rnd) >= 2:
            kt2_only.append(float(rnd[1]))
    mean_kt_round2 = float(np.mean(kt2_only)) if kt2_only else 0.0

    calib = mean_trunc_calib
    thr = overall_throughput
    composite = thr * (0.2 + 0.8 * calib) if total_time > 0 else 0.0

    mean_posthoc_gain_corrected = (
        float(np.mean([p[1] for p in posthoc_pairs_flat]))
        if posthoc_pairs_flat
        else 0.0
    )

    return {
        "all_accept_lengths": all_accept_lengths,
        "mean_accept_length": mean_accept,
        "response_lengths": response_lengths,
        "mean_response_length": mean_response_length,
        "total_response_length": total_response_length,
        "mean_per_turn_accept_length": mean_per_turn_accept,
        "mean_throughput": mean_throughput,
        "overall_throughput": overall_throughput,
        "target_total_time": target_total,
        "draft_total_time": draft_total,
        "total_time": total_time,
        "total_steps": total_steps,
        "mean_second_plus_kt": mean_second_plus_kt,
        "mean_kt_round2_only": mean_kt_round2,
        "mean_trunc_commit_le_accept_rate": mean_trunc_calib,
        "mean_inner_iters_reported": mean_inner_iters,
        "num_blocks_with_round2_plus": len(second_plus_all),
        "objective_composite": composite,
        "mean_posthoc_suffix_accept_gain": mean_posthoc_gain_corrected,
        "mean_posthoc_suffix_accept_gain_corrected": mean_posthoc_gain_corrected,
        "posthoc_suffix_gain_values": posthoc_gains_flat,
        "posthoc_suffix_refine_pairs": posthoc_pairs_flat,
        "n_posthoc_suffix_refine_executed": posthoc_executed_total,
        "n_posthoc_suffix_skipped_anchor_only": posthoc_skipped_anchor_total,
        "n_posthoc_suffix_skipped_acc1_out_of_range": posthoc_skipped_acc1_oor_total,
        "n_posthoc_suffix_events": len(posthoc_pairs_flat),
        "target_posthoc_extra_time": target_posthoc_extra,
    }


def _build_truncation_grid(
    policies: List[str],
    taus: List[float],
    risk_eps_list: List[float],
    inner_iters_list: List[int],
    use_tree_flags: List[bool],
) -> List[Dict[str, Any]]:
    rows = []
    for policy, iters, use_tree in product(policies, inner_iters_list, use_tree_flags):
        if policy == "full":
            rows.append(
                {
                    "truncation_policy": "full",
                    "max_block_inner_iters": iters,
                    "trunc_tau": 0.85,
                    "trunc_risk_eps": 0.35,
                    "use_draft_tree": use_tree,
                }
            )
        elif policy == "risk":
            for eps in risk_eps_list:
                rows.append(
                    {
                        "truncation_policy": "risk",
                        "max_block_inner_iters": iters,
                        "trunc_tau": 0.85,
                        "trunc_risk_eps": eps,
                        "use_draft_tree": use_tree,
                    }
                )
        else:
            for tau in taus:
                rows.append(
                    {
                        "truncation_policy": policy,
                        "max_block_inner_iters": iters,
                        "trunc_tau": tau,
                        "trunc_risk_eps": 0.35,
                        "use_draft_tree": use_tree,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description="DFlash++ / DFlash 投机解码测评")
    MODEL_PAIRS = {
        "qwen3-8b-o": {
            "target": "/share/public/public_models/Qwen3-8B",
            "draft": "z-lab/Qwen3-8B-DFlash-b16",
        },
        "qwen3-8b": {
            "target": "/share/public/public_models/Qwen3-8B",
            "draft": "/share/wanghanzhen/SpeculativeDecoding/NIPS26/dflash/model_weight/FlashMTP_v3.1",
        },
        "dflash-pp-trained": {
            "target": "/share/public/public_models/Qwen3-8B",
            "draft": _DEFAULT_DRAFT_PP,
        },
        "longwriter-llama3.1-8b": {
            "target": "THUDM/LongWriter-llama3.1-8b",
            "draft": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
        },
    }
    parser.add_argument(
        "--model-pair",
        type=str,
        default="dflash-pp-trained",
        choices=list(MODEL_PAIRS.keys()),
    )
    parser.add_argument("--target-model-path", type=str, default=None)
    parser.add_argument("--draft-model-path", type=str, default=None)
    parser.add_argument(
        "--question-file",
        type=str,
        default="/share/wanghanzhen/SpeculativeDecoding/NIPS26/dataset/mtbench101/question.jsonl",
    )
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="./model_answer")
    parser.add_argument("--debug-dir", type=str, default="./debug")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--draft-topk-file",
        type=str,
        default=None,
        help="draft top-k 行日志路径；默认关闭。截断 profile 建议 eval/trunck/draft_topk.jsonl，"
        "或与 --emit-trunc-profile 联用自动使用该路径。",
    )
    parser.add_argument(
        "--emit-trunc-profile",
        action="store_true",
        help=f"写入截断 profile 的 draft_topk 到 {_DEFAULT_TRUNC_PROFILE_DRAFT_TOPK}（若未再指定 --draft-topk-file）",
    )
    parser.add_argument("--draft-topk", type=int, default=2)
    parser.add_argument(
        "--max-block-inner-iters",
        type=int,
        default=2,
        help="块内迭代上限（≤3，与训练设计一致；默认 2）",
    )
    parser.add_argument(
        "--truncation-policy",
        type=str,
        default="min_prob",
        choices=["full", "min_prob", "risk", "score"],
    )
    parser.add_argument("--trunc-tau", type=float, default=0.88)
    parser.add_argument("--trunc-risk-eps", type=float, default=0.35)
    parser.add_argument("--trunc-score-beta4", type=float, default=0.15)
    parser.add_argument(
        "--inner-fallback",
        type=str,
        default="one",
        choices=["one", "none"],
    )
    parser.add_argument("--use-draft-tree", action="store_true")
    parser.add_argument("--draft-tree-branches", type=int, default=3)
    parser.add_argument(
        "--grid-truncation",
        action="store_true",
        help="对截断策略/迭代次数/草稿树做粗网格，输出 grid_truncation_results.json",
    )
    parser.add_argument(
        "--grid-include-tree",
        action="store_true",
        help="网格中包含 use_draft_tree=True（更慢）",
    )
    parser.add_argument(
        "--grid-max-new-tokens",
        type=int,
        default=2048,
        help="--grid-truncation 时每题生成的最大新 token（控时间）",
    )
    parser.add_argument(
        "--grid-preset",
        type=str,
        default="full",
        choices=["full", "fast"],
        help="fast：缩小网格，适合 test-in-loop；默认 full",
    )
    parser.add_argument(
        "--grid-sort-by",
        type=str,
        default="throughput",
        choices=["throughput", "accuracy", "composite"],
        help="accuracy=mean_trunc_commit_le_accept；composite=吞吐×校准加权",
    )
    parser.add_argument(
        "--spec-json",
        type=str,
        default=None,
        help="JSON 文件，字段并入 spec_generate_kw（覆盖 CLI 截断/迭代等）",
    )
    parser.add_argument(
        "--emit-run-summary-json",
        type=str,
        default=None,
        help="非 grid 模式下，将全部 turn 的汇总指标写入该路径",
    )
    parser.add_argument(
        "--record-posthoc-suffix-refine",
        action="store_true",
        help="后验实验：每验证步在已知 target 接受前缀后，对后缀再草稿一次并二次验证，"
        "统计 accept 长度增量（不改变真实解码轨迹；额外 target 计入手动字段）。",
    )
    parser.add_argument(
        "--posthoc-acc1-min",
        type=int,
        default=2,
        help="后验仅当首验 acc1 属于 [min,max] 且 acc1<块长时做二次迭代；<min 不迭代（默认 2）。",
    )
    parser.add_argument(
        "--posthoc-acc1-max",
        type=int,
        default=14,
        help="后验 acc1 上界（含）；>max 不二次迭代（默认 14）。",
    )
    args = parser.parse_args()

    if args.emit_trunc_profile and not args.draft_topk_file:
        args.draft_topk_file = _DEFAULT_TRUNC_PROFILE_DRAFT_TOPK

    if args.grid_truncation and args.output_dir == "./model_answer":
        args.output_dir = os.path.join(_TRUNCK_DIR, "grid_run")

    if args.target_model_path is None:
        args.target_model_path = MODEL_PAIRS[args.model_pair]["target"]
    if args.draft_model_path is None:
        args.draft_model_path = MODEL_PAIRS[args.model_pair]["draft"]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    target_model_name = os.path.basename(args.target_model_path.rstrip("/"))
    dataset_name = os.path.basename(os.path.dirname(args.question_file))
    tag = f"{'think' if args.thinking else ''}DFlashPP" if args.model_pair == "dflash-pp-trained" else f"{'think' if args.thinking else ''}DFlash"
    output_file = os.path.join(
        args.output_dir,
        f"{target_model_name}-{tag}-{dataset_name}-t{args.temperature}-ml{args.max_length}.jsonl",
    )
    log_file_path = os.path.join(
        args.output_dir,
        f"{target_model_name}-{tag}-{dataset_name}-t{args.temperature}-ml{args.max_length}.log",
    )

    register_local_dflash_model()
    draft_cfg = load_draft_qwen3_config(args.draft_model_path)
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model_path,
        config=draft_cfg,
        dtype="auto",
        device_map="cuda:0",
    ).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        dtype="auto",
        device_map="cuda:0",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    questions = load_mtbench101_questions(
        args.question_file, begin=args.begin, end=args.end
    )

    spec_kw_single = {
        "max_block_inner_iters": min(3, max(1, args.max_block_inner_iters)),
        "truncation_policy": args.truncation_policy,
        "trunc_tau": args.trunc_tau,
        "trunc_risk_eps": args.trunc_risk_eps,
        "trunc_score_beta4": args.trunc_score_beta4,
        "inner_fallback": args.inner_fallback,
        "use_draft_tree": args.use_draft_tree,
        "draft_tree_branches": args.draft_tree_branches,
        "record_posthoc_suffix_refine": bool(args.record_posthoc_suffix_refine),
        "posthoc_acc1_min": int(args.posthoc_acc1_min),
        "posthoc_acc1_max": int(args.posthoc_acc1_max),
    }
    if args.spec_json:
        with open(args.spec_json, "r", encoding="utf-8") as sf:
            spec_kw_single.update(json.load(sf))

    if args.grid_truncation:
        tree_flags = [False, True] if args.grid_include_tree else [False]
        if args.grid_preset == "fast":
            grid = _build_truncation_grid(
                policies=["full", "min_prob", "risk", "score"],
                taus=[0.88],
                risk_eps_list=[0.35, 0.5],
                inner_iters_list=[1, 3],
                use_tree_flags=[False],
            )
        else:
            grid = _build_truncation_grid(
                policies=["full", "min_prob", "risk", "score"],
                taus=[0.82, 0.88, 0.94],
                risk_eps_list=[0.28, 0.4, 0.55],
                inner_iters_list=[1, 2, 3],
                use_tree_flags=tree_flags,
            )
        grid_path = os.path.join(args.output_dir, "grid_truncation_results.json")
        grid_rows = []
        for gi, cfg in enumerate(grid):
            flat_turn_stats: List[dict] = []
            print(f"[grid {gi+1}/{len(grid)}] {cfg}", flush=True)
            for question in questions:
                draft_topk_fp = None
                try:
                    if args.draft_topk_file:
                        p = args.draft_topk_file + f".grid{gi}"
                        _d = os.path.dirname(os.path.abspath(p))
                        if _d:
                            os.makedirs(_d, exist_ok=True)
                        draft_topk_fp = open(p, "w", encoding="utf-8")
                    _, turn_stats = multi_turn_dialogue(
                        draft_model,
                        target_model,
                        tokenizer,
                        question["turns"],
                        max_new_tokens=min(args.max_length, args.grid_max_new_tokens),
                        temperature=args.temperature,
                        log_file=None,
                        thinking=args.thinking,
                        debug_dir=args.debug_dir if args.debug else None,
                        draft_topk_file=draft_topk_fp,
                        draft_topk=args.draft_topk,
                        spec_generate_kw=cfg,
                    )
                    flat_turn_stats.extend(turn_stats)
                finally:
                    if draft_topk_fp is not None:
                        draft_topk_fp.close()
            summ = summarize_question_stats(flat_turn_stats)
            grid_rows.append({"config": cfg, "summary": summ})
        if args.grid_sort_by == "accuracy":
            grid_rows.sort(
                key=lambda r: r["summary"].get(
                    "mean_trunc_commit_le_accept_rate", 0.0
                ),
                reverse=True,
            )
        elif args.grid_sort_by == "composite":
            grid_rows.sort(
                key=lambda r: r["summary"].get("objective_composite", 0.0),
                reverse=True,
            )
        else:
            grid_rows.sort(
                key=lambda r: r["summary"].get("overall_throughput", 0.0),
                reverse=True,
            )
        with open(grid_path, "w", encoding="utf-8") as gf:
            json.dump(grid_rows, gf, ensure_ascii=False, indent=2)
        top = grid_rows[0]
        print(f"Grid done (sort={args.grid_sort_by}). Top config: {top['config']}", flush=True)
        print(
            f"  throughput={top['summary']['overall_throughput']:.2f} tok/s | "
            f"calib={top['summary']['mean_trunc_commit_le_accept_rate']:.3f} | "
            f"composite={top['summary'].get('objective_composite', 0):.2f}",
            flush=True,
        )
        return

    draft_topk_fp = None
    if args.draft_topk_file:
        ddir = os.path.dirname(os.path.abspath(args.draft_topk_file))
        if ddir:
            os.makedirs(ddir, exist_ok=True)
        draft_topk_fp = open(args.draft_topk_file, "w", encoding="utf-8")

    all_turn_stats_run: List[dict] = []
    try:
        with open(output_file, "w", encoding="utf-8") as f_out, open(
            log_file_path, "w", encoding="utf-8"
        ) as f_log:
            f_log.write(f"Loaded {len(questions)} questions from {args.question_file}\n")
            f_log.write(f"Target Model: {args.target_model_path}\n")
            f_log.write(f"Draft Model: {args.draft_model_path}\n")
            f_log.write(f"spec_generate_kw: {json.dumps(spec_kw_single, ensure_ascii=False)}\n\n")

            for idx, question in enumerate(questions):
                f_log.write("\n" + "=" * 80 + "\n")
                f_log.write(
                    f"Question {idx + 1}/{len(questions)} | ID: {question['question_id']}\n"
                )
                f_log.flush()
                responses, turn_stats = multi_turn_dialogue(
                    draft_model,
                    target_model,
                    tokenizer,
                    question["turns"],
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    log_file=f_log,
                    thinking=args.thinking,
                    debug_dir=args.debug_dir if args.debug else None,
                    draft_topk_file=draft_topk_fp,
                    draft_topk=args.draft_topk,
                    spec_generate_kw=spec_kw_single,
                )
                summary = summarize_question_stats(turn_stats, responses)
                f_log.write("-" * 80 + "\nQuestion Summary\n")
                f_log.write(f"mean_second_plus_kt: {summary['mean_second_plus_kt']:.4f}\n")
                f_log.write(f"mean_kt_round2_only: {summary['mean_kt_round2_only']:.4f}\n")
                f_log.write(
                    f"mean_trunc_commit_le_accept_rate: "
                    f"{summary['mean_trunc_commit_le_accept_rate']:.4f}\n"
                )
                f_log.write(f"Overall Mean Accept Length: {summary['mean_accept_length']:.4f}\n")
                f_log.write(f"Overall Throughput: {summary['overall_throughput']:.2f} tokens/sec\n")
                f_log.write("-" * 80 + "\n")
                f_log.flush()
                result = {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "turns": question["turns"],
                    "responses": responses,
                    "statistics": {"turn_stats": turn_stats, "summary": summary},
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                all_turn_stats_run.extend(turn_stats)
            run_summary = summarize_question_stats(all_turn_stats_run)
            f_log.write("\n" + "=" * 80 + "\nDATASET RUN SUMMARY\n")
            f_log.write(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n")
            f_log.write(f"\nResults saved to: {output_file}\n")
            if args.emit_run_summary_json:
                with open(args.emit_run_summary_json, "w", encoding="utf-8") as ej:
                    json.dump(run_summary, ej, ensure_ascii=False, indent=2)
    finally:
        if draft_topk_fp is not None:
            draft_topk_fp.close()


if __name__ == "__main__":
    main()
