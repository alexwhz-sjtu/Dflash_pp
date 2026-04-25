
# DFlash++（dflash_pp）项目级进展

本文档汇总本仓库中 **DFlash++** 推理/测评/训练设计相关动机、技术点、成败与实验状态；细节命令与基线数字另见 `progress.md`、`iterative_block_generation_design.md`。

---

# 一、动机

- **块内一次性预测整段**（长度 `B`）时，越靠后的位置缺乏「已确认前缀」的自条件，与 target 按**最长连续前缀**验收的机制不匹配，容易造成后半段浪费（接收长度常明显小于 `B`）。
- **目标**：在保持 target 侧 KV 注入与 DFlash 条件方式一致的前提下，通过**块内多轮去噪**（迭代扩展已确认前缀 + 重填后缀）、**截断/置信启发**与可选**草稿树**，提高有效接收长度与端到端吞吐。
- **训练侧方向**（见 `iterative_block_generation_design.md`）：从「一步猜全块」转为「给定任意干净前缀，预测剩余后缀」；可叠加连续性损失 `L_con` 等。

---

# 二、推理流程

## 1. 截断点选取算法

- **实现位置**：`specforge/modeling/draft/dflash.py` 中 `_suffix_metrics`（`pmax` / `margin` / `ent`）与 `_truncate_suffix_length`。
- **策略**：`full`（不截断）、`min_prob`（首个 `pmax < τ`）、`risk`（累计 `1-pmax` 超预算）、`score`（`log pmax + margin - ent - β·位置惩罚` 与阈值比较）。
- **inner_fallback**：与 target 是否接受无关；`one` 表示规则截出 `k=0` 时仍强制块内再取 1 个 suffix token，避免整轮不推进。纯截断分析可用 `none`。
- **后验校准脚本**：`eval/calibrate_truncation_posthoc.py`，默认产出在 **`eval/trunck/`**（JSON + Markdown 报告）。历史副本可能在 `eval/_til_out/calibration_truncation_report.md`。
- **已知现象**：多数场景下，块内除 anchor 外 top1 置信度随位置大致衰减；被拒位置置信度往往较低（详见校准报告与 top2 表）。

## 2. 自适应渐进式迭代次数设计

- **参数**：`max_block_inner_iters`（上限 3，与训练设计对齐），`truncation_policy` 与各 `trunc_tau` / `risk_eps` / `score_beta`。
- **行为**：每轮对已确认前缀保留 token，后缀置 mask，草稿前向后再按策略截断提交若干位置；`full` + `iters=1` 等价于单轮整块 argmax 填满再验。
- **测评**：`eval/eval.py` 统计 `inner_kt_rounds`、`mean_second_plus_kt`、`mean_kt_round2_only` 等。

## 3. 草稿树设计

- **实现**：`draft_tree_block`——探针 + 在弱置信附近取若干 cut，分支并行 refine，按 mean log pmax 选枝。
- **事故与修复**：扩充 `cuts` 的 `while` 在 `max(cuts)==block_size-1` 时若无法加入新元素会导致 **CPU 死循环**（GPU 0%）；已改为双向尝试并无进展则 `break`。
- **实测**：`eval/test_in_loop.py` 汇总见 `eval/_til_out/test_in_loop_report.json`；当前配置下树侧草稿前向更多，端到端吞吐 **低于** 同设置下最优线稿（见报告内 `draft_tree_speedup_vs_best_line`）。

---

# 三、训练设计

## 1. 训练目标

- 详见 **`iterative_block_generation_design.md`**：块内状态 `z^{(t)}`（干净前缀 + mask 后缀）、条件为 target 融合 hidden；损失可含原有 DFlash 块内 CE 与连续性/迭代对齐相关项。

## 2. L_con 干净前缀与推理路径对齐

- **结构**：`scripts/train_dflash_pp.py` 为训练入口，超参在 `scripts/run_training_dflash_pp.sh` 中注入；`specforge/core/dflash_pp.py` 的 `OnlineDFlashPPModel` 将 **L_dflash** 与 **L_con** 合并在**一次** draft 前向中完成：沿 batch 维拼接「标准 DFlash 块噪声」与「completion 噪声」，对两半 logits 分别计算损失。
- **L_con 块内状态构造**：用 `_sample_prefix_lengths` 得到每个锚点块上的干净前缀长度 `p`，再经 `_create_noise_embed_completion` 在块内把 **第 `p` 个及之后** 的位置换成 mask、之前位置用真值 token 嵌入。即 L_con 学的是「给定已对齐的**一段**前缀，补全余下后缀」。
- **与推理一致的约束**：干净前缀采样的**边界**不落在块内最前三格。块内位置 **0（anchor）**、**1**、**2** 在 L_con 分支中**始终**为干净真值，等价于只从 `p ∈ {3,…,B-1}` 中按原有 `w,b` 的 softmax 分布采样「mask 起点」；0/1/2 不会被选成「从该位起全 mask 后缀」的切分点。
- **动机**：后验实验（见下节 §四.1）表明，**第一次** target 接受长度 **≤2** 时，再做多一轮块内去噪**几乎无**接受长度增量；故推理上规定该情况**不**进入第二次迭代。训练侧将 L_con 的随机前缀与之一致，避免强监督大量「首段极短、推理从不走第二轮」的补全模式。


---

# 四、相关实验

## 1. 动机验证

验证训练所得草稿是否能在「单次块内迭代」得到的接收长度基础上，通过**额外迭代**可拓展 target 连续接受长度。

### 1.1 后验性的验证渐进式去噪有效性

**协议（已实现，不改变真实解码轨迹）**：

1. 与线上一致：`max_block_inner_iters=1`（或用户指定）、草稿填满块（如 `truncation_policy=full`），target 做第一次并行验证，得到接受长度 `acc1`（最长连续匹配，含实现中的 +1 约定）。
2. 若 `acc1 < B`：将块内前 `acc1` 个 token 视为**已对齐干净前缀**，后缀重新置 mask；在**同一 target 上下文 KV** 下快照、恢复（见下）后，对后缀做 **一次** 草稿前向并重填，再对**整块**做第二次 target 验证，得到 `acc2`。
3. 记录 **`acc2 - acc1`** 作为该验证步上「多一轮去噪」带来的接受长度增量；全序列写回仍按**第一次**验证结果，保证与常规投机解码一致。

**KV cache**：在第一次 `target(block)` 前快照 `past_key_values_target` 与 `past_key_values_draft`（refine 后）；第一次验证后另存 target KV；二次验证仅用于计数，随后 **restore** 第一次验证后的 target KV 与 refine 后的 draft KV，因此主路径与不计 posthoc 时一致。

**代码**：`spec_generate(..., record_posthoc_suffix_refine=True)`；CLI：`eval/eval.py --record-posthoc-suffix-refine`。

**数据**：

- 硬性设定为迭代两次时：`mean_posthoc_suffix_accept_gain ≈ 0.65`；
- 发现**第一次**接受长度 **≤2** 的块上，**再**迭代去噪**几乎无**收益，故推理侧规定该情况不进入第二次块内迭代；**训练**侧 L_con 的干净前缀长度仅采样 `p ≥ 3`（块内前三格恒干净），与上述决策对齐（见 **§三.2**）。
- 在应用该推理规则后，后验度量的 `mean_posthoc_suffix_accept_gain` 可升至约 `1.29` 量级（依具体脚本与子集而定）。

**复现**：


【状态】**已完成（代码 + 示例数据）**；更大 `begin/end`、不同 `truncation_policy` 可继续扫 `eval/trunck/`。

## 2. 技术点实验

### 2.1 截断点选取算法验证

见 **第四节 1** 与 `eval/trunck/`、`eval/calibrate_truncation_posthoc.py`。后续可试窗口 min、log 概率积累计阈值等（见 `project_progess.md` 历史讨论）。

【状态】**初步探索**

### 2.2 自适应迭代次数

部分块首 token 即被拒可少迭代；一次去噪已接受过半则边际小——可与截断置信、`max_block_inner_iters` 网格联调（`eval/eval.py --grid-truncation`，默认网格产出目录见 `eval/trunck/grid_run` 等）。

【状态】**待系统化**

### 2.3 训练部分

见 **`iterative_block_generation_design.md`** 与 `scripts/run_training_dflash_pp.sh`。

【状态】**设计文档已有；训练消融按规划推进**

---

# 五、环境与路径约定

- 推荐 **`source .venv/bin/activate`** 后运行 `eval/` 脚本。
- **截断 profile / 后验实验产出** 默认建议目录：**`eval/trunck/`**（含 `draft_topk`、`posthoc_*`、`calibration_*`、`grid_fast` 等）。
- `test_in_loop` 的 smoke 与总报告仍可能在 **`eval/_til_out/`**。
