#!/bin/bash
# FlashMTP 训练启动脚本

set -e

# 自动激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# ========================================
# 配置参数
# ========================================

# GPU 设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 目标模型路径
TARGET_MODEL="${TARGET_MODEL:-$WHZ_DIR/models/Qwen/Qwen3-8B}"

# 训练参数（MAX_LENGTH 需要在数据路径构建之前定义）
NUM_EPOCHS="${NUM_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 数据特征参数（用于自动构建数据路径）
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:-600000}"
ENABLE_THINKING="${ENABLE_THINKING:-on}"

# 构建数据子目录名: n{N|all}_think_{on|off}
DATASET_BASE_DIR="${DATASET_BASE_DIR:-./cache/dataset}"
if [ "${ENABLE_THINKING}" = "on" ] || [ "${ENABLE_THINKING}" = "true" ] || [ "${ENABLE_THINKING}" = "1" ]; then
    THINK_STR="on"
else
    THINK_STR="off"
fi
DATA_SUBDIR="n${DATA_NUM_SAMPLES}_think_${THINK_STR}"

# 数据目录（支持通过 TRAIN_DATA_PATH 直接指定，否则自动构建）
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATASET_BASE_DIR}/${DATA_SUBDIR}/train_regen.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./cache/models/flashmtp_${DATA_SUBDIR}_maxlen${MAX_LENGTH}}"
CACHE_DIR="${CACHE_DIR:-${DATASET_BASE_DIR}/${DATA_SUBDIR}}"

# 模型参数
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
DRAFT_HIDDEN_SIZE="${DRAFT_HIDDEN_SIZE:-}"
DRAFT_INTERMEDIATE_SIZE="${DRAFT_INTERMEDIATE_SIZE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
CONCAT_MODE="${CONCAT_MODE:-seq}"  # "seq" 或 "feature"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"

# 日志和保存间隔
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"

# Tracker 参数
REPORT_TO="${REPORT_TO:-none}"  # none, wandb, tensorboard
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp-training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# 分布式参数
TP_SIZE="${TP_SIZE:-1}"
DIST_TIMEOUT="${DIST_TIMEOUT:-30}"

# 数据参数
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
IS_PREFORMATTED="${IS_PREFORMATTED:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
BUILD_DATASET_NUM_PROC="${BUILD_DATASET_NUM_PROC:-8}"

# 恢复训练
RESUME="${RESUME:-}"
CKPT_DIR="${CKPT_DIR:-}"

# ========================================
# 显示配置
# ========================================
echo "=========================================="
echo "FlashMTP 训练启动脚本"
echo "=========================================="
echo "数据特征:"
echo "  样本数量: ${DATA_NUM_SAMPLES}"
echo "  思考模式: ${THINK_STR}"
echo "  数据子目录: ${DATA_SUBDIR}"
echo "------------------------------------------"
echo "目标模型: ${TARGET_MODEL}"
echo "训练数据: ${TRAIN_DATA_PATH}"
echo "评估数据: ${EVAL_DATA_PATH:-无}"
echo "输出目录: ${OUTPUT_DIR}"
echo "缓存目录: ${CACHE_DIR}"
echo "------------------------------------------"
echo "模型配置:"
echo "  草稿模型层数: ${NUM_DRAFT_LAYERS}"
echo "  草稿模型hidden_size: ${DRAFT_HIDDEN_SIZE:-使用目标模型配置}"
echo "  草稿模型intermediate_size: ${DRAFT_INTERMEDIATE_SIZE:-使用目标模型配置}"
echo "  块大小: ${BLOCK_SIZE}"
echo "  锚点数量: ${NUM_ANCHORS}"
echo "  拼接模式: ${CONCAT_MODE}"
echo "  Attention后端: ${ATTENTION_BACKEND}"
echo "------------------------------------------"
echo "训练配置:"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  批大小: ${BATCH_SIZE} x ${ACCUMULATION_STEPS} = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  学习率: ${LEARNING_RATE}"
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

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}

# ========================================
# 训练
# ========================================
echo ""
echo "==> 开始训练 FlashMTP"
echo ""

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")
else
    LAUNCHER=(python)
fi

# 构建可选参数
OPTIONAL_ARGS=""

if [ -n "${EVAL_DATA_PATH}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --eval-data-path ${EVAL_DATA_PATH}"
fi

if [ -n "${DRAFT_HIDDEN_SIZE}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --draft-hidden-size ${DRAFT_HIDDEN_SIZE}"
fi

if [ -n "${DRAFT_INTERMEDIATE_SIZE}" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --draft-intermediate-size ${DRAFT_INTERMEDIATE_SIZE}"
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

if [ "${REPORT_TO}" != "none" ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --report-to ${REPORT_TO}"
    if [ "${REPORT_TO}" = "wandb" ] && [ -n "${WANDB_PROJECT}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-project ${WANDB_PROJECT}"
    fi
    if [ -n "${WANDB_RUN_NAME}" ]; then
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --wandb-run-name ${WANDB_RUN_NAME}"
    fi
fi

"${LAUNCHER[@]}"    ./scripts/train_flashmtp.py \
    --target-model-path ${TARGET_MODEL} \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir ${OUTPUT_DIR} \
    --cache-dir ${CACHE_DIR} \
    --num-draft-layers ${NUM_DRAFT_LAYERS} \
    --block-size ${BLOCK_SIZE} \
    --num-anchors ${NUM_ANCHORS} \
    --concat-mode ${CONCAT_MODE} \
    --attention-backend ${ATTENTION_BACKEND} \
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

# ========================================
# 训练完成
# ========================================
echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: ${OUTPUT_DIR}"
echo ""
echo "使用示例："
echo "  from specforge.modeling.draft.flashmtp import FlashMTPDraftModel"
echo "  draft_model = FlashMTPDraftModel.from_pretrained('${OUTPUT_DIR}/epoch_6_step_<step>')"
echo ""
echo "运行推理："
echo "  python benchmark.py --draft-model ${OUTPUT_DIR}/epoch_6_step_<step>"
echo "=========================================="
