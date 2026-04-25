#!/usr/bin/env bash
# DFlash++ 测评入口：与训练脚本中的 LCON_MIN_PREFIX_LEN 对齐，传入 eval/eval.py
#
# 用法:
#   export LCON_MIN_PREFIX_LEN=3   # 默认 3；与 run_training_dflash_pp.sh 中变量同名
#   ./scripts/run_eval_dflash_pp.sh
# 或: LCON_MIN_PREFIX_LEN=4 ./scripts/run_eval_dflash_pp.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

LCON_MIN_PREFIX_LEN="${LCON_MIN_PREFIX_LEN:-3}"
# 后验下界不指定时由 eval 侧用 --lcon-min-prefix-len；若需与 K 不同可设 POSTHOC_ACC1_MIN
POSTHOC_ACC1_MIN="${POSTHOC_ACC1_MIN:-}"

cd "${PROJECT_DIR}"
EXTRA=()
if [ -n "${POSTHOC_ACC1_MIN}" ]; then
  EXTRA+=(--posthoc-acc1-min "${POSTHOC_ACC1_MIN}")
fi

echo "LCON_MIN_PREFIX_LEN=${LCON_MIN_PREFIX_LEN}  (与训练 run_training_dflash_pp.sh 一致时勿改偏)"
if [ -n "${POSTHOC_ACC1_MIN}" ]; then
  echo "POSTHOC_ACC1_MIN=${POSTHOC_ACC1_MIN} (覆盖后验下界，默认与 K 相同)"
fi

python eval/eval.py \
  --lcon-min-prefix-len "${LCON_MIN_PREFIX_LEN}" \
  "${EXTRA[@]}" \
  "$@"
