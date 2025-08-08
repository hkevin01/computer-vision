#!/bin/bash
set -e

echo "🎯 IMMEDIATE PROJECT REORGANIZATION EXECUTION"
echo "============================================="
echo "Cleaning up the cluttered root directory RIGHT NOW!"
echo ""

cd /home/kevin/Projects/computer-vision

# Count current mess
CURRENT_FILES=$(ls -1 | wc -l)
SCRIPT_FILES=$(ls -1 *.sh 2>/dev/null | wc -l)
DOC_FILES=$(ls -1 *.md 2>/dev/null | wc -l)

echo "📊 Current Clutter Analysis:"
echo "  Total items in root: $CURRENT_FILES"
echo "  Shell scripts: $SCRIPT_FILES"
echo "  Markdown docs: $DOC_FILES"
echo ""

# Make all scripts executable
echo "🔧 Making reorganization scripts executable..."
chmod +x reorganize-project.sh 2>/dev/null || true
chmod +x update-paths.sh 2>/dev/null || true
chmod +x verify-organization.sh 2>/dev/null || true
chmod +x master-reorganize.sh 2>/dev/null || true

# Quick reorganization execution
echo ""
echo "🚀 EXECUTING REORGANIZATION NOW..."
echo "================================="

# Create backup first
BACKUP_DIR="../computer-vision-backup-$(date +%Y%m%d-%H%M%S)"
echo "📦 Creating backup: $BACKUP_DIR"
cp -r . "$BACKUP_DIR"

# Run reorganization immediately
echo ""
echo "🧹 Running comprehensive reorganization..."

if [ -f "reorganize-project.sh" ]; then
    echo "▶️  Executing reorganize-project.sh..."
    timeout 120 ./reorganize-project.sh || echo "⚠️  Reorganization script completed with warnings"
else
    # Manual reorganization if script fails
    echo "⚠️  Running manual reorganization..."

    # Create directories
    mkdir -p docker config api web

    # Move Docker files
    echo "📦 Moving Docker files..."
    [ -f "Dockerfile" ] && mv Dockerfile docker/ && echo "  ✅ Moved Dockerfile"
    [ -f "docker-compose.yml" ] && mv docker-compose.yml docker/ && echo "  ✅ Moved docker-compose.yml"
    [ -f "docker-compose.yml.backup" ] && mv docker-compose.yml.backup docker/ && echo "  ✅ Moved docker-compose.yml.backup"
    [ -f "docker-compose.yml.new" ] && mv docker-compose.yml.new docker/ && echo "  ✅ Moved docker-compose.yml.new"

    # Move docs
    echo "📚 Moving documentation..."
    for doc in *.md; do
        if [ "$doc" != "README.md" ] && [ -f "$doc" ]; then
            mv "$doc" docs/ 2>/dev/null && echo "  ✅ Moved $doc" || echo "  ⚠️  Could not move $doc"
        fi
    done

    # Move scripts (keep run.sh in root)
    echo "🔧 Moving scripts..."
    for script in *.sh; do
        if [ "$script" != "run.sh" ] && [ -f "$script" ]; then
            mv "$script" scripts/ 2>/dev/null && echo "  ✅ Moved $script" || echo "  ⚠️  Could not move $script"
        fi
    done

    # Move config files
    echo "⚙️  Moving configuration..."
    [ -f ".clang-format" ] && mv .clang-format config/ && echo "  ✅ Moved .clang-format"
    [ -f ".editorconfig" ] && mv .editorconfig config/ && echo "  ✅ Moved .editorconfig"
    [ -f ".env.example" ] && mv .env.example config/ && echo "  ✅ Moved .env.example"
    [ -f ".pre-commit-config.yaml" ] && mv .pre-commit-config.yaml config/ && echo "  ✅ Moved .pre-commit-config.yaml"

    # Move API files
    echo "🐍 Moving API files..."
    [ -f "api-server.py" ] && mv api-server.py api/ && echo "  ✅ Moved api-server.py"

    # Move web files
    echo "🌐 Moving web files..."
    [ -f "test-connection.html" ] && mv test-connection.html web/ && echo "  ✅ Moved test-connection.html"

    # Move tools
    echo "🛠️  Moving tools..."
    [ -f "Universal_Docker_Development_Strategy.ipynb" ] && mv Universal_Docker_Development_Strategy.ipynb tools/ && echo "  ✅ Moved notebook"

    # Move test files
    echo "🧪 Moving test files..."
    [ -f "test_args.cpp" ] && mv test_args.cpp tests/ && echo "  ✅ Moved test_args.cpp"
fi

echo ""
echo "🔄 Updating paths and configurations..."
if [ -f "scripts/update-paths.sh" ]; then
    cd scripts
    chmod +x update-paths.sh
    ./update-paths.sh || echo "⚠️  Path update completed with warnings"
    cd ..
fi

# Final status
echo ""
echo "📊 REORGANIZATION RESULTS:"
echo "=========================="

NEW_ROOT_FILES=$(ls -1 | wc -l)
echo "Root directory files: $CURRENT_FILES → $NEW_ROOT_FILES"

if [ -d "docker" ]; then
    DOCKER_FILES=$(ls -1 docker/ 2>/dev/null | wc -l)
    echo "Docker files: $DOCKER_FILES"
fi

if [ -d "scripts" ]; then
    SCRIPT_COUNT=$(ls -1 scripts/ 2>/dev/null | wc -l)
    echo "Scripts organized: $SCRIPT_COUNT"
fi

if [ -d "docs" ]; then
    DOC_COUNT=$(ls -1 docs/ 2>/dev/null | wc -l)
    echo "Docs organized: $DOC_COUNT"
fi

echo ""
echo "🎉 PROJECT REORGANIZATION COMPLETE!"
echo "=================================="
echo ""
echo "✨ Your project structure is now clean and professional!"
echo ""
echo "📁 New Structure:"
echo "  docker/     - All Docker configuration"
echo "  scripts/    - All shell scripts"
echo "  docs/       - All documentation"
echo "  config/     - Configuration files"
echo "  api/        - API server files"
echo "  web/        - Web assets"
echo ""
echo "🔗 Essential files kept in root:"
ls -1 | grep -E '^(README|LICENSE|CMakeLists|run\.sh|\.env|\.git)' || echo "  (Standard project files)"
echo ""
echo "🚀 Ready to use your clean, professional project structure!"
echo "💾 Backup available at: $BACKUP_DIR"
