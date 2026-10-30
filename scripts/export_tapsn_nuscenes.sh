#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/export_tapsn_nuscenes.sh [--start-town TOWN] [--only-town TOWN] [OUTPUT_DIR]

Exports all TAPSN/Town*/<sequence>/sampling sequences into one nuScenes-style
dataset by calling utils/nuscenes_converter/export_nuscenes.py.

Environment:
  DATA_ROOT   TAPSN root. Default: ./TAPSN
  OUTPUT_DIR  Output dataset root. Default: <DATA_ROOT>_nuscenes
  VERSION     nuScenes metadata version. Default: v1.0-trainval
  PYTHON_BIN  Python executable. Default: python3
USAGE
}

START_TOWN=""
ONLY_TOWN=""
OUTPUT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --start-town)
      if [[ $# -lt 2 ]]; then
        echo "--start-town requires a town name" >&2
        exit 1
      fi
      START_TOWN="$2"
      shift 2
      ;;
    --only-town)
      if [[ $# -lt 2 ]]; then
        echo "--only-town requires a town name" >&2
        exit 1
      fi
      ONLY_TOWN="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "${OUTPUT_ARG}" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 1
      fi
      OUTPUT_ARG="$1"
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  if [[ -n "${OUTPUT_ARG}" || $# -gt 1 ]]; then
    echo "Unexpected extra argument: $*" >&2
    usage >&2
    exit 1
  fi
  OUTPUT_ARG="$1"
fi

if [[ -n "${START_TOWN}" && -n "${ONLY_TOWN}" ]]; then
  echo "--start-town and --only-town cannot be used together" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-./TAPSN}"
OUTPUT_DIR="${OUTPUT_ARG:-${OUTPUT_DIR:-${DATA_ROOT}_nuscenes}}"
VERSION="${VERSION:-v1.0-trainval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPORTER="${REPO_ROOT}/utils/nuscenes_converter/export_nuscenes.py"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${EXPORTER}" ]]; then
  echo "Exporter does not exist: ${EXPORTER}" >&2
  exit 1
fi

mapfile -t LABEL_FILES < <(
  find "${DATA_ROOT}" \
    -mindepth 4 \
    -maxdepth 4 \
    -type f \
    -path "${DATA_ROOT}/Town*/*/sampling/labels.json" \
    | sort
)

if [[ -n "${START_TOWN}" || -n "${ONLY_TOWN}" ]]; then
  SELECTED_LABEL_FILES=()
  for label_file in "${LABEL_FILES[@]}"; do
    sequence_dir="$(dirname "$(dirname "${label_file}")")"
    town_name="$(basename "$(dirname "${sequence_dir}")")"
    if [[ -n "${ONLY_TOWN}" && "${town_name}" != "${ONLY_TOWN}" ]]; then
      continue
    fi
    if [[ -n "${START_TOWN}" && "${town_name}" < "${START_TOWN}" ]]; then
      continue
    fi
    SELECTED_LABEL_FILES+=("${label_file}")
  done
  LABEL_FILES=("${SELECTED_LABEL_FILES[@]}")
fi

if (( ${#LABEL_FILES[@]} == 0 )); then
  echo "No sequence labels found under ${DATA_ROOT}/Town*/<sequence>/sampling/labels.json" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "VERSION: ${VERSION}"
echo "START_TOWN: ${START_TOWN:-}"
echo "ONLY_TOWN: ${ONLY_TOWN:-}"
echo "SEQUENCES: ${#LABEL_FILES[@]}"

for label_file in "${LABEL_FILES[@]}"; do
  sequence_dir="$(dirname "$(dirname "${label_file}")")"
  echo "Exporting ${sequence_dir}"
  "${PYTHON_BIN}" "${EXPORTER}" \
    "${sequence_dir}" \
    "${OUTPUT_DIR}" \
    --version "${VERSION}" \
    --append \
    --replace-sequence
done

echo "Done."
