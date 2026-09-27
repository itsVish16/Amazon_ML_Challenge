#!/usr/bin/env bash
set -euo pipefail

# Project and Data Directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DATA/student_resource/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output}"

# Tuning Parameters
BATCH_SIZE="${BATCH_SIZE:-2500}"
MAX_CANDIDATES="${MAX_CANDIDATES:-25}"
NUM_WORKERS="${NUM_WORKERS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Cap workers to 48 on 64 vCPU hosts for sustained high-throughput inference
if [ "${NUM_WORKERS}" -gt 48 ]; then
  NUM_WORKERS=48
elif [ "${NUM_WORKERS}" -gt 4 ]; then
  NUM_WORKERS=$((NUM_WORKERS - 2))
fi

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo " Launching High-Throughput Entity Resolution Runner"
echo " Host CPUs:        $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 'Unknown')"
echo " Worker Processes: ${NUM_WORKERS}"
echo " Batch Size:       ${BATCH_SIZE}"
echo " Test Dir:         ${DATA_DIR}/test"
echo " Output Dir:       ${OUTPUT_DIR}"
echo "=================================================="

exec python3 -u "${PROJECT_DIR}/code/business_entity_resolution/src/pipeline.py" \
  --test-dir "${DATA_DIR}/test" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --num-workers "${NUM_WORKERS}"
