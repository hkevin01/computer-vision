#!/bin/bash
set -e

echo "🧹 COMPREHENSIVE PROJECT REORGANIZATION (subset)"
echo "=============================================="
echo "This script performs a subset of the reorganization focused on reproducible moves."
echo ""

cd "$(git rev-parse --show-toplevel || pwd)"

mkdir -p docker config documentation

echo "Moving selected docker and docs files into docker/ and docs/"
for file in Dockerfile docker-compose.yml DOCKER_RUNNER_README.md DOCKER_SETUP_COMPLETE.md; do
    if [ -f "$file" ]; then
        mv "$file" docker/ || true
        echo "Moved $file -> docker/"
    fi
done

echo "Done subset reorganize"
