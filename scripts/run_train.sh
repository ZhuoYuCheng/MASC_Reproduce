#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
RUN_TAG="${1:-unsup}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/${RUN_TAG}_${STAMP}.log"

TMUX_SESSION="masc_${RUN_TAG}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}"
  echo "Attach with: tmux attach -t ${TMUX_SESSION}"
  exit 1
fi

CMD="/root/miniconda3/bin/python train.py --mode ${RUN_TAG}"
echo "Starting: ${CMD}"
tmux new-session -d -s "${TMUX_SESSION}" "cd /root/autodl-tmp/MASC_Reproduce && ${CMD} 2>&1 | tee ${LOG}"
echo "tmux session: ${TMUX_SESSION}"
echo "log file: ${LOG}"
