#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR=../data/3d_fishes
OUTPUT_DIR=../data/3d_fishes

WIDTH=1024
HEIGHT=1024

mkdir -p "$OUTPUT_DIR"

for src in "$INPUT_DIR"/*.glb; do
  base=$(basename "$src" .glb)
  dst="$OUTPUT_DIR/${base}_resized.glb"
  echo "▶ Resizing $base.glb → $(basename "$dst")"
  gltf-transform resize \
    "$src" \
    "$dst" \
    --width "$WIDTH" \
    --height "$HEIGHT"
done

echo "✅ All done! Resized files are in $OUTPUT_DIR"
