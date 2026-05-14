#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCES_DIR="$ROOT_DIR/resources"

force_full=0
force_quantized=0
force_ivf=0

for arg in "$@"; do
  case "$arg" in
    --force-full)
      force_full=1
      ;;
    --force-quantized)
      force_quantized=1
      ;;
    --force-ivf)
      force_ivf=1
      ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "usage: $0 [--force-full] [--force-quantized] [--force-ivf]" >&2
      exit 1
      ;;
  esac
done

exists() {
  [[ -f "$1" ]]
}

newer_than() {
  local source="$1"
  local target="$2"
  [[ -f "$source" && -f "$target" && "$source" -nt "$target" ]]
}

run_step() {
  local label="$1"
  local script_path="$2"

  echo "==> $label"
  python3 "$ROOT_DIR/$script_path"
}

references_json="$RESOURCES_DIR/references.json"
vectors_npy="$RESOURCES_DIR/vectors.npy"
vectors_f16_npy="$RESOURCES_DIR/vectors_f16.npy"
vectors_q8_npy="$RESOURCES_DIR/vectors_q8.npy"
tree_npz="$RESOURCES_DIR/tree.npz"
labels_npy="$RESOURCES_DIR/labels.npy"
norms_npy="$RESOURCES_DIR/norms.npy"
centroids_npy="$RESOURCES_DIR/centroids.npy"
cluster_indices_npy="$RESOURCES_DIR/cluster_indices.npy"
cluster_offsets_npy="$RESOURCES_DIR/cluster_offsets.npy"

need_index=0
need_quantized=0
need_ivf=0

if (( force_full )); then
  need_index=1
else
  if ! exists "$vectors_npy" || ! exists "$labels_npy" || ! exists "$norms_npy"; then
    need_index=1
  elif exists "$references_json" && newer_than "$references_json" "$vectors_npy"; then
    need_index=1
  fi
fi

if (( need_index )); then
  need_quantized=0
  need_ivf=1
elif (( force_quantized )); then
  need_quantized=1
else
  if ! exists "$vectors_f16_npy" || ! exists "$vectors_q8_npy" || ! exists "$tree_npz"; then
    need_quantized=1
  elif newer_than "$vectors_npy" "$vectors_f16_npy" || newer_than "$vectors_npy" "$vectors_q8_npy"; then
    need_quantized=1
  elif newer_than "$labels_npy" "$tree_npz" || newer_than "$vectors_npy" "$tree_npz"; then
    need_quantized=1
  fi
fi

if (( force_ivf )); then
  need_ivf=1
elif ! (( need_ivf )); then
  if ! exists "$centroids_npy" || ! exists "$cluster_indices_npy" || ! exists "$cluster_offsets_npy"; then
    need_ivf=1
  elif newer_than "$vectors_npy" "$centroids_npy" || newer_than "$labels_npy" "$centroids_npy"; then
    need_ivf=1
  elif newer_than "$vectors_f16_npy" "$centroids_npy" || newer_than "$vectors_q8_npy" "$centroids_npy"; then
    need_ivf=1
  fi
fi

if (( need_index )); then
  run_step "building base index from references.json" "scripts/build_index.py"
fi

if (( need_quantized )); then
  run_step "building quantized vectors and tree" "scripts/build_quantized.py"
fi

if (( need_ivf )); then
  run_step "building ivf index" "scripts/build_ivf.py"
fi

if ! (( need_index || need_quantized || need_ivf )); then
  echo "==> resources are already up to date"
fi
