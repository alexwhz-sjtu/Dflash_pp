#!/bin/bash
# DFlash++ 训练启动脚本（与 run_training_dflash.sh 对齐，入口为 train_dflash_pp.py）

set -e

DT="a800"
# posttraining：从 DFlash 权重继续训，默认较低学习率；scratch：从零训练，沿用原较大学习率
MODE="posttraining"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dt) DT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [[ "$DT" != "qz" && "$DT" != "a800" ]]; then
    echo "错误: --dt 须为 qz 或 a800"
    exit 1
fi
if [[ "$MODE" != "posttraining" && "$MODE" != "scratch" ]]; then
    echo "错误: --mode 须为 posttraining 或 scratch"
    exit 1
fi

expory

# a800：更保守锚点数，减轻显存峰值
if [ "$DT" = "a800" ]; then
    MAX_LENGTH="${MAX_LENGTH:-4096}"
    NUM_ANCHORS="${NUM_ANCHORS:-512}"
else
    MAX_LENGTH="${MAX_LENGTH:-4096}"
    NUM_ANCHORS="${NUM_ANCHORS:-512}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi


# ========================================
# 主要训练参数
# ========================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

NUM_EPOCHS="${NUM_EPOCHS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
# 按模式默认学习率（仍可用环境变量 LEARNING_RATE 覆盖）
if [ "$MODE" = "scratch" ]; then
    LEARNING_RATE="${LEARNING_RATE:-6e-4}"
else
    LEARNING_RATE="${LEARNING_RATE:-3e-4}"
fi
MAX_LENGTH="${MAX_LENGTH:-4096}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"


# ========================================
# 主要数据集参数
# ========================================
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-400000}"
ENABLE_THINKING="${ENABLE_THINKING:-on}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-./cache/dataset}"
if [ "${ENABLE_THINKING}" = "on" ] || [ "${ENABLE_THINKING}" = "true" ] || [ "${ENABLE_THINKING}" = "1" ]; then
    THINK_STR="on"
else
    THINK_STR="off"
fi
DATA_SUBDIR="n${DATA_NUM_SAMPLES}_think_${THINK_STR}"


EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
CACHE_DIR="${CACHE_DIR:-./cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}}"

NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-7}"

# DFlash++：总损失 = μ*L_dflash + λ*L_con；前缀 p 采样 logits_p = -w*(p-1-b)^2
DFLASH_LOSS_WEIGHT="${DFLASH_LOSS_WEIGHT:-0.4}"
COMPLETION_LOSS_WEIGHT="${COMPLETION_LOSS_WEIGHT:-0.6}"
COMPLETION_GAMMA="${COMPLETION_GAMMA:-7}"
COMPLETION_PREFIX_SAMPLE_WEIGHT="${COMPLETION_PREFIX_SAMPLE_WEIGHT:-0.1}"
COMPLETION_PREFIX_SAMPLE_BIAS="${COMPLETION_PREFIX_SAMPLE_BIAS:-0.0}"

# 默认不做「训练集前 N 条」eval；单独评估请设 EVAL_DATA_PATH。若需恢复可 export EVAL_TRAIN_MAX_SAMPLES>0
EVAL_TRAIN_MAX_SAMPLES="${EVAL_TRAIN_MAX_SAMPLES:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"

# ========================================
# 主要路径
# ========================================
if [ "$DT" = "qz" ]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/nemotron_${DATA_NUM_SAMPLES}/nemotron_think_${ENABLE_THINKING}_samples_${DATA_NUM_SAMPLES}_qwen3_8b_regen.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/DFlash_pp_sample_${DATA_NUM_SAMPLES}_think_${ENABLE_THINKING}_qwen3_8b_maxlen${MAX_LENGTH}_epochs${NUM_EPOCHS}}"
    TARGET_MODEL="${TARGET_MODEL:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/Qwen/Qwen3-8B}"
else
    TRAIN_DATA_PATH="/share/wanghanzhen/SpeculativeDecoding/NIPS26/FlashMTP_v1.1/cache/data/regen_data/nemotron_40000/nemotron_think_on_samples_40000_qwen3_8b_regen.jsonl"
    OUTPUT_DIR="./cache/models/flashmtp_v3.1_nemotron_think_on_samples_40000_qwen3_8b"
    TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
fi

# scratch 默认不加载 DFlash；posttraining 默认加载预训练 DFlash 草稿
if [ "$MODE" = "posttraining" ]; then
    if [ "$DT" = "qz" ]; then
        INIT_DRAFT_FROM="${INIT_DRAFT_FROM:-/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/models/z-lab/Qwen3-8B-DFlash-b16}"
    else
        INIT_DRAFT_FROM="${INIT_DRAFT_FROM:-/share/wanghanzhen/.cache/huggingface/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/071541888480df12d8a1ef7acbaabed88b0a8bd4}"
    fi
else
    INIT_DRAFT_FROM="${INIT_DRAFT_FROM:-}"
fi

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-hf}"

LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-dflash_pp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_DIR="${WANDB_DIR:-./wandb}"
WANDB_RUN_ID="${WANDB_RUN_ID:-dflash_pp_sample_${DATA_NUM_SAMPLES}_lbase_${DFLASH_LOSS_WEIGHT}_lcon_${COMPLETION_LOSS_WEIGHT}}"

TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-30}"

CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3-thinking}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"



echo "=========================================="
echo "DFlash++ 训练启动脚本"
echo "  模式: ${MODE} (posttraining=从DFlash继续训+低LR, scratch=从零+原LR)"
echo "=========================================="
echo "数据特征:"
echo "  样本数量: ${DATA_NUM_SAMPLES}"
echo "  思考模式: ${THINK_STR}"
echo "  数据子目录: ${DATA_SUBDIR}"
echo "------------------------------------------"
echo "目标模型: ${TARGET_MODEL}"
echo "目标模型后端: ${TARGET_MODEL_BACKEND}"
echo "训练数据: ${TRAIN_DATA_PATH}"
echo "评估数据: ${EVAL_DATA_PATH:-无}"
echo "输出目录: ${OUTPUT_DIR}"
echo "缓存目录: ${CACHE_DIR}"
echo "------------------------------------------"
echo "模型配置:"
echo "  草稿模型层数: ${NUM_DRAFT_LAYERS}"
echo "  块大小: ${BLOCK_SIZE}"
echo "  锚点数量: ${NUM_ANCHORS}"
echo "  Attention后端: ${ATTENTION_BACKEND}"
echo "  DFlash Loss衰减Gamma: ${LOSS_DECAY_GAMMA:-未设置(不启用)}"
echo "  L_con 权重 λ: ${COMPLETION_LOSS_WEIGHT}"
echo "  L_con 权重 γ: ${COMPLETION_GAMMA}"
echo "  L_con 前缀采样 w,b: ${COMPLETION_PREFIX_SAMPLE_WEIGHT}, ${COMPLETION_PREFIX_SAMPLE_BIAS}"
echo "  L_dflash 权重 μ: ${DFLASH_LOSS_WEIGHT}"
echo "  init-draft-from: ${INIT_DRAFT_FROM:-未设置}"
echo "------------------------------------------"
echo "训练配置:"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  批大小: ${BATCH_SIZE} x ${ACCUMULATION_STEPS} = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  学习率: ${LEARNING_RATE}"
echo "  eval 间隔: ${EVAL_INTERVAL} (train子集eval: ${EVAL_TRAIN_MAX_SAMPLES} 条, 0=关闭; 有 EVAL_DATA_PATH 时用独立 eval; 每轮最多 ${EVAL_MAX_BATCHES} batch)"
echo "  最大长度: ${MAX_LENGTH}"
echo "  预热比例: ${WARMUP_RATIO}"
echo "  梯度裁剪: ${MAX_GRAD_NORM}"
echo "------------------------------------------"
echo "分布式配置:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  TP_SIZE: ${TP_SIZE}"
echo "------------------------------------------"
echo "Tracker: ${REPORT_TO}"
echo "=========================================="
echo ""

original_output_dir="${OUTPUT_DIR}"
suffix=1
while [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; do
    OUTPUT_DIR="${original_output_dir}_${suffix}"
    suffix=$((suffix + 1))
done
if [ "${OUTPUT_DIR}" != "${original_output_dir}" ]; then
    echo "警告: 输出目录 ${original_output_dir} 已存在且非空，自动切换到: ${OUTPUT_DIR}"
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

echo ""
echo "==> 开始训练 DFlash++"
echo ""

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")
else
    LAUNCHER=(python)
fi

OPTIONAL_ARGS=""

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval-data-path ${EVAL_DATA_PATH}"
fi

if [ -n "${LOSS_DECAY_GAMMA}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --loss-decay-gamma ${LOSS_DECAY_GAMMA}"
fi

if [ -n "${IS_PREFORMATTED}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --is-preformatted"
fi

if [ -n "${RESUME}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --resume"
fi

if [ -n "${CKPT_DIR}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --ckpt-dir ${CKPT_DIR}"
fi

if [ -n "${INIT_DRAFT_FROM:-}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --init-draft-from ${INIT_DRAFT_FROM}"
fi

if [ "${REPORT_TO}" != "none" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --report-to ${REPORT_TO}"
    if [ "${REPORT_TO}" = "wandb" ] && [ -n "${WANDB_PROJECT}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-project ${WANDB_PROJECT}"
    fi
    if [ -n "${WANDB_RUN_NAME}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-run-name ${WANDB_RUN_NAME}"
    fi
fi

"${LAUNCHER[@]}"    ./scripts/train_dflash_pp.py \
    --target-model-path ${TARGET_MODEL} \
    --target-model-backend ${TARGET_MODEL_BACKEND} \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir ${OUTPUT_DIR} \
    --cache-dir ${CACHE_DIR} \
    --num-draft-layers ${NUM_DRAFT_LAYERS} \
    --block-size ${BLOCK_SIZE} \
    --num-anchors ${NUM_ANCHORS} \
    --attention-backend ${ATTENTION_BACKEND} \
    --completion-loss-weight ${COMPLETION_LOSS_WEIGHT} \
    --completion-gamma ${COMPLETION_GAMMA} \
    --completion-prefix-sample-weight ${COMPLETION_PREFIX_SAMPLE_WEIGHT} \
    --completion-prefix-sample-bias ${COMPLETION_PREFIX_SAMPLE_BIAS} \
    --dflash-loss-weight ${DFLASH_LOSS_WEIGHT} \
    --eval-train-max-samples ${EVAL_TRAIN_MAX_SAMPLES} \
    --eval-max-batches ${EVAL_MAX_BATCHES} \
    --training-mode ${MODE} \
    --learning-rate ${LEARNING_RATE} \
    --warmup-ratio ${WARMUP_RATIO} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --accumulation-steps ${ACCUMULATION_STEPS} \
    --max-grad-norm ${MAX_GRAD_NORM} \
    --max-length ${MAX_LENGTH} \
    --log-interval ${LOG_INTERVAL} \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --chat-template ${CHAT_TEMPLATE} \
    --dataloader-num-workers ${DATALOADER_NUM_WORKERS} \
    --build-dataset-num-proc ${BUILD_DATASET_NUM_PROC} \
    --tp-size ${TP_SIZE} \
    --dist-timeout ${DIST_TIMEOUT} \
    --seed 42 \
    ${OPTIONAL_ARGS}

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: ${OUTPUT_DIR}"
echo ""
echo "使用示例："
echo "  from specforge.modeling.draft.dflash import DFlashDraftModel"
echo "  draft_model = DFlashDraftModel.from_pretrained('${OUTPUT_DIR}/epoch_6_step_<step>')"
echo "=========================================="
