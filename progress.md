# DFlash++ 推理与测评进度

## 运行环境

- **uv 虚拟环境（推荐）**：`source .venv/bin/activate` 后运行 `eval/` 下脚本。
- **Conda**：若使用 **`eagle`** 环境，例如：
  - `conda activate eagle`
  - 再进入仓库根目录运行，例如：
  - `python eval/eval.py --begin 0 --end 2 --max-length 512`（冒烟）
  - `python eval/eval.py --grid-truncation --begin 0 --end 1 --grid-max-new-tokens 512`（粗网格，输出 `model_answer/grid_truncation_results.json`）
  - 草稿树：`python eval/eval.py --use-draft-tree` 或网格加 `--grid-include-tree`

## 总目标

1. 在 `specforge/modeling/draft/dflash.py` 中完成 **DFlash++ 块内迭代** 的 `spec_generate`（最多 3 轮块内迭代 + 可配置置信度截断）。
2. 在 `eval/eval.py` 上扩展 **完整测评**：迭代收益统计、截断策略校准与后验准确度、端到端速度网格搜索、可选 **草稿树** 与加速比。
3. 草稿权重默认：`cache/models/flashmtp_v3.1_nemotron_think_on_samples_40000_qwen3_8b/epoch_6_step_12750`。

## 执行计划（Phase 1 → Phase 2）

| 阶段 | 内容 | 状态 |
|------|------|------|
| P1-A | `spec_generate`：块内迭代、`min_prob` / `risk` / `score` 截断、`inner_fallback`、计时与 `get_last_decode_stats` | 已完成 |
| P1-B | `eval.py`：`specforge` 导入、默认 `dflash-pp-trained` 权重、CLI、`summarize` 扩展、`--grid-truncation` | 已完成 |
| P1-C | 跑通短子集 sanity（mtbench101 1 题、ml512） | **已完成**（`eval/_til_out/smoke_summary.json`） |
| P2 | 草稿树 A/B 与吞吐比 | **已实测**：`eval/_til_out/test_in_loop_report.json`（本配置下树侧草稿前向更多，端到端 **低于** 最优线稿吞吐，见报告内 `draft_tree_speedup_vs_best_line`） |

## 基线：块内只迭代 1 次（`max_block_inner_iters=1`）

**设定**：`truncation_policy=full`（单轮前向整块 Argmax 填满再验，无置信度截断）；mtbench101 `question.jsonl` **`--begin 0 --end 1`**（1 条多轮题，共 3 个 assistant turn）；`--max-length 512`；草稿权重为默认 `dflash-pp-trained`。完整 JSON：`eval/_til_out/baseline_inner1_summary.json`，日志与逐题：`eval/_til_out/baseline_inner1/`。

| 指标 | 数值 |
|------|------|
| 端到端吞吐 `overall_throughput` | **97.28** tok/s |
| 平均每验证步接受长 `mean_accept_length` | **4.22** |
| 平均每 turn 回复长 `mean_response_length` | **428.33** tokens |
| 本 run 总生成 token `total_response_length` | **1285** |
| 总墙钟 `total_time` | **13.21** s |
| Target / Draft 时间 | **11.25** / **1.96** s |
| Target 验证步数 `total_steps` | **308** |
| 块内第 2 轮及以后 `k` 之和（均值）`mean_second_plus_kt` | **0**（符合仅 1 轮迭代） |
| 仅第 2 轮 `k` 均值 `mean_kt_round2_only` | **0** |
| 平均每块块内迭代轮数 `mean_inner_iters_reported` | **1.0** |
| 截断提交≤实际接受 后验命中率 `mean_trunc_commit_le_accept_rate` | **0.0134** |

复现命令：

```bash
source .venv/bin/activate
python eval/eval.py --model-pair dflash-pp-trained \
  --question-file /share/wanghanzhen/SpeculativeDecoding/NIPS26/dataset/mtbench101/question.jsonl \
  --begin 0 --end 1 --max-length 512 \
  --max-block-inner-iters 1 --truncation-policy full \
  --output-dir eval/_til_out/baseline_inner1 \
  --emit-run-summary-json eval/_til_out/baseline_inner1_summary.json
```

## 指标说明（与用户需求对齐）

- **第二次预测增加的接收长度**：对每个 target 验证步，记录块内迭代各轮 `k_t`；报告第 2、3 轮 `k_t` 的分布及与仅 1 轮块内迭代的对照（eval 可用 `--compare-inner-iters` 或网格中 iters=1 vs 3）。
- **截断策略**：`min_prob`（连缀阈值）、`risk`（累计风险预算）、`score`（log pmax + margin − 熵 − 位置惩罚）；后验记录 `commit_len` vs 实际 `accept_len`（相对块起点），校准命中率。
- **端到端速度**：`overall_throughput`、draft/target 时间占比；`--grid-truncation` 对 `(policy, tau, max_block_inner_iters)` 粗搜索。
- **草稿树**：在 Phase 2 测量相对「无树 + 最优线性截断」的加速比。

## 变更文件

- `specforge/modeling/draft/dflash.py` — `spec_generate` 与统计
- `eval/eval.py` — 测评入口与聚合

## 日志

- 2026-04-22：初始化计划；开始实现 P1-A。
- 2026-04-22：`spec_generate` 与 `eval/eval.py` 联调完成；`max_block_inner_iters` 在推理侧强制 cap 为 3；网格结果写入 `output-dir/grid_truncation_results.json`。
- 2026-04-22：**草稿树** `draft_tree_block` 中扩充 `cuts` 的 `while` 在 `max(cuts)==block_size-1` 时无法增加元素导致 **CPU 死循环**（GPU 0%）；已改为双向尝试扩展并无进展则 `break`。测评前请 `source .venv/bin/activate`。
- 2026-04-22：修复后重跑 `ab_tree`（512 tokens），汇总 **`eval/_til_out/test_in_loop_report.json`**（含 smoke / grid_fast_top5 / 最优线稿 vs 草稿树吞吐与比值）。
- 2026-04-22：基线 **`max_block_inner_iters=1` + `full`**，1 题 ml512，结果表见上文；汇总 `eval/_til_out/baseline_inner1_summary.json`。
