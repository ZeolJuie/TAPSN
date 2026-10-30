#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MAPS=("Town02_Opt")

for MAP in "${MAPS[@]}"; do
  echo "--------------------------------------------"
  echo "Processing: $MAP"

  TARGET_DIR="${REPO_ROOT}/topology/$MAP"
  mkdir -p "$TARGET_DIR"

  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    conda run -n carla python "${REPO_ROOT}/utils/generate_walkable_cells.py" \
      --map-name "$MAP" \
      --output-dir "$TARGET_DIR"
done

echo "All Done!"
