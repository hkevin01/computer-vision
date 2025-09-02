#!/bin/bash
set -e

echo "🧹 EXECUTING PROJECT REORGANIZATION"
echo "==================================="
echo "Transforming cluttered root directory into clean structure"
echo ""

cd /home/kevin/Projects/computer-vision

# Make all scripts executable first
echo "🔧 Making scripts executable..."
chmod +x *.sh 2>/dev/null || true

# Count files in root before reorganization
echo ""
echo "📊 BEFORE REORGANIZATION:"
echo "Files in root directory: $(ls -1 | grep -v '^[a-z]' | wc -l)"
echo "Total items in root: $(ls -1 | wc -l)"

# Show some of the clutter
echo ""
echo "🗂️  Current root directory clutter (sample):"
ls -1 | grep -E '\.(md|sh|yml|py)$' | head -10 | sed 's/^/  - /'
if [ $(ls -1 | grep -E '\.(md|sh|yml|py)$' | wc -l) -gt 10 ]; then
    echo "  ... and $(( $(ls -1 | grep -E '\.(md|sh|yml|py)$' | wc -l) - 10 )) more files"
fi

echo ""
echo "🚀 Starting reorganization process..."
echo ""

# Execute the master reorganization script
if [ -f "master-reorganize.sh" ]; then
    echo "▶️  Executing master-reorganize.sh..."
    ./master-reorganize.sh
else
    echo "❌ master-reorganize.sh not found, running individual steps..."

    # Fallback to running individual scripts
    if [ -f "reorganize-project.sh" ]; then
        echo "▶️  Running reorganize-project.sh..."
        ./reorganize-project.sh
    fi

    if [ -f "update-paths.sh" ]; then
        echo "▶️  Running update-paths.sh..."
        ./update-paths.sh
    fi

    if [ -f "verify-organization.sh" ]; then
        echo "▶️  Running verify-organization.sh..."
        ./verify-organization.sh
    fi
fi

echo ""
echo "🎉 REORGANIZATION EXECUTION COMPLETE!"
echo "===================================="
