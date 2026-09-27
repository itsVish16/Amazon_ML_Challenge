#!/usr/bin/env bash
set -euo pipefail

# The data directory is intentionally external to Git.  Override these paths
# when the challenge data is mounted elsewhere on the instance.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DATA/student_resource/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
MAX_CANDIDATES="${MAX_CANDIDATES:-25}"

mkdir -p "${OUTPUT_DIR}"
exec python3 -u "${PROJECT_DIR}/code/business_entity_resolution/src/pipeline.py" \
  --test-dir "${DATA_DIR}/test" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-candidates "${MAX_CANDIDATES}"
