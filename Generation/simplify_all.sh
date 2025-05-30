#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="../data/3d_fishes"
OUTPUT_DIR="../data/3d_fishes"

mkdir -p "${OUTPUT_DIR}"


for input_path in "${INPUT_DIR}"/*.glb; do
  filename="$(basename "${input_path}")"
  base="${filename%.glb}"
  output_path="${OUTPUT_DIR}/${base}3.glb"

  echo "Processing: ${filename} -> ${base}3.glb"
  gltf-transform simplify \
    "${input_path}" \
    "${output_path}" \
    --error 0.01
done

echo "All done! Simplified files are in ${OUTPUT_DIR}"
