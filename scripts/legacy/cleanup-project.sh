#!/bin/bash

# Project Root Cleanup Script
# Organizes documentation files while maintaining functionality

set -e

echo "🧹 Starting Project Root Cleanup..."

# Create directories if they don't exist
mkdir -p docs/setup
mkdir -p docs/planning
mkdir -p docs/process

echo "📁 Moving documentation files..."

# Planning documents (move to docs/planning/)
planning_files=(
    "docs/planning/AI_ML_IMPROVEMENTS_SUMMARY.md"
    "docs/architectural/IMPLEMENTATION_PLAN.md"
    "docs/planning/IMPROVEMENTS_ROADMAP.md"
    "OPENCV_OPTIMIZATION.md"
    "PROJECT_MODERNIZATION_STRATEGY.md"
)

for file in "${planning_files[@]}"; do
    if [[ -f "$file" ]]; then
        if [[ -s "$file" ]]; then  # Only move if file has content
            echo "  Moving $file → docs/planning/"
            mv "$file" "docs/planning/"
        else
            echo "  Removing empty $file"
            rm "$file"
        fi
    fi
done

# Process documents (move to docs/process/)
process_files=(
    "DIRECTORY_CLEANUP_SUMMARY.md"
    "WORKFLOW.md"
)

for file in "${process_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  Moving $file → docs/process/"
        mv "$file" "docs/process/"
    fi
done

# Setup documents are already moved
echo "  ✅ Docker setup files already organized in docs/setup/"

echo ""
echo "🔗 Creating convenient symlinks..."

# Create symlinks for frequently accessed files
cd docs/setup
if [[ ! -L "../../DOCKER_SETUP.md" ]]; then
    ln -sf docs/setup/docker-setup.md ../../DOCKER_SETUP.md
    echo "  Created DOCKER_SETUP.md → docs/setup/docker-setup.md"
fi

if [[ ! -L "../../QUICK_START.md" ]]; then
    ln -sf docs/setup/docker-readme.md ../../QUICK_START.md
    echo "  Created QUICK_START.md → docs/setup/docker-readme.md"
fi

cd ../..

echo ""
echo "📊 Root directory contents after cleanup:"
ls -la | grep -E '\.(md|yml|yaml|sh|txt)$|^[A-Z]' | head -20

echo ""
echo "✅ Project root cleanup completed!"
echo ""
echo "🎯 Quick access:"
echo "   ./run.sh                    - Main build/run script"
echo "   ./DOCKER_SETUP.md          - Docker setup guide (→ docs/setup/)"
echo "   ./QUICK_START.md           - Quick start guide (→ docs/setup/)"
echo "   ./docs/                    - All organized documentation"
echo ""
