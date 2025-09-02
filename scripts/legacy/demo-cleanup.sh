#!/bin/bash

echo "🎯 DEMONSTRATION: CLEANING UP CLUTTERED PROJECT"
echo "=============================================="
echo ""

cd /home/kevin/Projects/computer-vision

# Show current clutter
echo "📊 BEFORE CLEANUP:"
echo "Root directory files: $(ls -1 | wc -l)"
echo "Sample clutter:"
ls -1 | grep -E '\.(sh|md|yml)$' | head -8 | sed 's/^/  /'
echo ""

# Make cleanup script executable
chmod +x clean-now.sh

echo "🚀 EXECUTING CLEANUP..."
echo "======================"
echo ""

# Run the cleanup
./clean-now.sh

echo ""
echo "🔍 VERIFICATION:"
echo "==============="

# Show results
echo "📊 AFTER CLEANUP:"
echo "Root directory files: $(ls -1 | wc -l)"
echo ""
echo "🗂️  Remaining in root:"
ls -1 | sed 's/^/  /'

echo ""
echo "📁 Organized structure:"
for dir in docker config api web docs scripts tools; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  $dir/: $count files"
    fi
done

echo ""
echo "✨ REORGANIZATION DEMONSTRATION COMPLETE!"
echo "Your project is now clean and professional! 🎉"
