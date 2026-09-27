#!/usr/bin/env bash
set -euo pipefail

# Project and Data Directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DATA/student_resource/dataset}"
SRC_DIR="${PROJECT_DIR}/code/business_entity_resolution/src"

echo "=================================================="
echo " Full-Target Hard-Negative Training"
echo " Data Dir:  ${DATA_DIR}/train"
echo " Source:    ${SRC_DIR}"
echo "=================================================="

exec python3 -u "${SRC_DIR}/train.py" \
  --data-dir "${DATA_DIR}/train" \
  --s1-samples 50000 \
  --full-targets \
  --seed 20260927
