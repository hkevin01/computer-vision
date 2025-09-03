#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"

CONF="$ROOT_DIR/config/models_urls.sh"
if [ -f "$CONF" ]; then
  # shellcheck disable=SC1090
  source "$CONF"
fi

echo "Fetching models into $MODELS_DIR"

download() {
  url="$1"
  out="$2"
  sha256="$3"
  echo "Downloading $url -> $out"
  curl -L --fail --progress-bar -o "$out" "$url"
  if [ -n "$sha256" ]; then
    echo "$sha256  $out" | sha256sum -c -
  fi
}

if [ -n "$HITNET_URL" ]; then
  download "$HITNET_URL" "$MODELS_DIR/hitnet.onnx" "$HITNET_SHA"
else
  echo "HITNet URL not configured, skipping hitnet";
fi

if [ -n "$RAFT_STEREO_URL" ]; then
  download "$RAFT_STEREO_URL" "$MODELS_DIR/raft_stereo.onnx" "$RAFT_STEREO_SHA"
else
  echo "RAFT_STEREO_URL not configured, skipping raft_stereo";
fi

if [ -n "$CRESTEREO_URL" ]; then
  download "$CRESTEREO_URL" "$MODELS_DIR/crestereo.onnx" "$CRESTEREO_SHA"
else
  echo "CRESTEREO_URL not configured, skipping crestereo";
fi

echo "Models fetch complete."
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
