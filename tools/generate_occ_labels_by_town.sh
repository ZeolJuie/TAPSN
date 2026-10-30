#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/generate_occ_labels_by_town.sh TOWN [MAX_JOBS] [-- EXTRA_GENERATE_ARGS...]

Examples:
  tools/generate_occ_labels_by_town.sh Town01_Opt
  tools/generate_occ_labels_by_town.sh Town03_Opt 8 -- --len_sequence 100

Environment:
  TAPSN_ROOT   Input dataset root. Default: ./TAPSN
  GTS_ROOT     Output gts root. Default: ./gts
  PYTHON       Python executable. Default: python
  CONFIG_PATH  Occupancy config path. Default: <repo>/config/occ/default.toml
  SENSOR_NAME  Semantic LiDAR sensor name. Default: LIDAR_TOP_SEMANTIC
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOWN_INPUT="$1"
shift

MAX_JOBS="${MAX_JOBS:-8}"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    MAX_JOBS="$1"
    shift
fi
if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer: ${MAX_JOBS}" >&2
    exit 2
fi

if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
fi
EXTRA_ARGS=("$@")

TAPSN_ROOT="${TAPSN_ROOT:-./TAPSN}"
GTS_ROOT="${GTS_ROOT:-./gts}"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/occ/default.toml}"
SENSOR_NAME="${SENSOR_NAME:-LIDAR_TOP_SEMANTIC}"

if [[ -d "${TAPSN_ROOT}/${TOWN_INPUT}" ]]; then
    TOWN_NAME="${TOWN_INPUT}"
else
    echo "Town directory not found: ${TAPSN_ROOT}/${TOWN_INPUT}" >&2
    exit 1
fi

if [[ "${TOWN_NAME}" =~ ^Town0*([0-9]+) ]]; then
    printf -v MAP_ID "%02d" "${BASH_REMATCH[1]}"
else
    echo "Cannot parse map id from town name: ${TOWN_NAME}" >&2
    exit 1
fi

TOWN_DIR="${TAPSN_ROOT}/${TOWN_NAME}"
GENERATE_SCRIPT="${REPO_ROOT}/utils/generate_occ_labels.py"

if [[ ! -f "${GENERATE_SCRIPT}" ]]; then
    echo "generate_occ_labels.py not found: ${GENERATE_SCRIPT}" >&2
    exit 1
fi

SEQUENCE_DIRS=()
while IFS= read -r -d '' seq_dir; do
    if [[ -f "${seq_dir}/sampling/labels.json" ]]; then
        SEQUENCE_DIRS+=("${seq_dir}")
    fi
done < <(find "${TOWN_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ ${#SEQUENCE_DIRS[@]} -eq 0 ]]; then
    echo "No sequence directories with sampling/labels.json found under ${TOWN_DIR}" >&2
    exit 1
fi

mkdir -p "${GTS_ROOT}"

run_one() {
    local seq_dir="$1"
    local seq_id="$2"
    local scene_name="$3"
    local out_path="${GTS_ROOT}/${scene_name}"
    local log_path="${out_path}/generate_occ_labels.log"

    mkdir -p "${out_path}"
    printf '[%s] start: %s\n' "${scene_name}" "${seq_dir}"

    if "${PYTHON_BIN}" "${GENERATE_SCRIPT}" \
        --data_path "${seq_dir}" \
        --out_path "${out_path}" \
        --config_path "${CONFIG_PATH}" \
        --sensor_name "${SENSOR_NAME}" \
        "${EXTRA_ARGS[@]}" >"${log_path}" 2>&1; then
        printf '[%s] done: %s\n' "${scene_name}" "${out_path}"
        return 0
    fi

    local status=$?
    printf '[%s] failed (%d), log: %s\n' "${scene_name}" "${status}" "${log_path}" >&2
    return "${status}"
}

echo "Town: ${TOWN_NAME}"
echo "Input: ${TOWN_DIR}"
echo "Output root: ${GTS_ROOT}"
echo "Sequences: ${#SEQUENCE_DIRS[@]}"
echo "Max jobs: ${MAX_JOBS}"

failures=0
running=0

for seq_dir in "${SEQUENCE_DIRS[@]}"; do
    seq_id="$(basename "${seq_dir}")"
    scene_name="scene-${MAP_ID}-${seq_id}"

    run_one "${seq_dir}" "${seq_id}" "${scene_name}" &
    ((running+=1))

    if (( running >= MAX_JOBS )); then
        if ! wait -n; then
            ((failures+=1))
        fi
        ((running-=1))
    fi
done

while (( running > 0 )); do
    if ! wait -n; then
        ((failures+=1))
    fi
    ((running-=1))
done

if (( failures > 0 )); then
    echo "Finished with ${failures} failed sequence(s)." >&2
    exit 1
fi

echo "All sequences finished."
