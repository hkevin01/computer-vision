#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"

declare -A MODELS
MODELS[hitnet]="https://example.com/models/hitnet.onnx"
MODELS[raft_stereo]="https://example.com/models/raft_stereo.onnx"
MODELS[crestereo]="https://example.com/models/crestereo.onnx"

echo "Fetching models into $MODELS_DIR"
for name in "${!MODELS[@]}"; do
  url=${MODELS[$name]}
  dest="$MODELS_DIR/${name}.onnx"
  if [ -f "$dest" ]; then
    echo "$dest already exists, skipping"
    continue
  fi
  echo "Downloading $name from $url"
  curl -L --fail -o "$dest" "$url"
done

echo "Note: replace example.com URLs with real model URLs or update config/models.yaml"
