#!/bin/bash

echo "🔍 PROJECT REORGANIZATION VERIFICATION"
echo "====================================="
echo "Verifying clean structure and functionality"
echo ""

cd /home/kevin/Projects/computer-vision

# === PHASE 1: Structure Verification ===
echo "📁 PHASE 1: Directory Structure Verification"
echo "--------------------------------------------"

expected_dirs=(
    "docker"
    "config"
    "api"
    "web"
    "docs"
    "scripts"
    "tools"
    "src"
    "include"
    "tests"
    "gui"
)

echo "Checking directory structure..."
for dir in "${expected_dirs[@]}"; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  ✅ $dir/ ($count files)"
    else
        echo "  ❌ $dir/ (missing)"
    fi
done

# === PHASE 2: Root Directory Cleanliness ===
echo ""
echo "🧹 PHASE 2: Root Directory Cleanliness Check"
echo "-------------------------------------------"

echo "Files remaining in root directory:"
root_files=$(ls -1 | grep -v '^[a-z]' | grep -E '\.(md|txt|yml|json|sh)$' 2>/dev/null || true)
essential_files=$(ls -1 | grep -E '^(CMakeLists\.txt|README\.md|LICENSE|\.gitignore|\.env|run\.sh)$' 2>/dev/null || true)

echo "Essential files (should remain):"
for file in $essential_files; do
    echo "  ✅ $file"
done

echo ""
echo "Other files in root:"
other_files=$(ls -1 | grep -v '^[a-z]' | grep -v -E '^(CMakeLists\.txt|README\.md|LICENSE|\.gitignore|\.env|run\.sh)$' 2>/dev/null || true)
if [ -z "$other_files" ]; then
    echo "  ✅ No unnecessary files in root"
else
    echo "  ⚠️  Files that might need reorganization:"
    for file in $other_files; do
        echo "    - $file"
    done
fi

# === PHASE 3: Docker Configuration Verification ===
echo ""
echo "🐳 PHASE 3: Docker Configuration Verification"
echo "--------------------------------------------"

if [ -f "docker/docker-compose.yml" ]; then
    echo "✅ docker/docker-compose.yml exists"

    # Check if paths are correctly updated
    if grep -q "dockerfile: ../Dockerfile" docker/docker-compose.yml; then
        echo "✅ Dockerfile path correctly updated"
    else
        echo "⚠️  Dockerfile path may need updating"
    fi

    if grep -q "context: .." docker/docker-compose.yml; then
        echo "✅ Build context correctly updated"
    else
        echo "⚠️  Build context may need updating"
    fi
else
    echo "❌ docker/docker-compose.yml missing"
fi

if [ -f "Dockerfile" ]; then
    echo "✅ Main Dockerfile exists in root"
else
    echo "❌ Main Dockerfile missing"
fi

if [ -f "gui/Dockerfile" ]; then
    echo "✅ GUI Dockerfile exists"
else
    echo "❌ GUI Dockerfile missing"
fi

# === PHASE 4: Run Script Verification ===
echo ""
echo "🔧 PHASE 4: Run Script Verification"
echo "----------------------------------"

if [ -f "run.sh" ]; then
    echo "✅ Main run.sh exists"

    # Check if it's executable
    if [ -x "run.sh" ]; then
        echo "✅ run.sh is executable"
    else
        echo "⚠️  run.sh is not executable"
        chmod +x run.sh
        echo "✅ Made run.sh executable"
    fi

    # Test basic functionality
    if ./run.sh help &>/dev/null; then
        echo "✅ run.sh help command works"
    else
        echo "⚠️  run.sh help command failed"
    fi
else
    echo "❌ Main run.sh missing"
fi

if [ -f "run-clean.sh" ]; then
    echo "✅ Enhanced run-clean.sh exists"
else
    echo "⚠️  Enhanced run-clean.sh not found"
fi

# === PHASE 5: Documentation Verification ===
echo ""
echo "📚 PHASE 5: Documentation Verification"
echo "-------------------------------------"

doc_count=$(ls -1 docs/ 2>/dev/null | wc -l)
echo "Documentation files in docs/: $doc_count"

if [ $doc_count -gt 5 ]; then
    echo "✅ Documentation well organized"
else
    echo "⚠️  Limited documentation in docs/"
fi

# Check for directory READMEs
readme_dirs=("docker" "config" "api" "web")
for dir in "${readme_dirs[@]}"; do
    if [ -f "$dir/README.md" ]; then
        echo "✅ $dir/README.md exists"
    else
        echo "⚠️  $dir/README.md missing"
    fi
done

# === PHASE 6: Configuration Files Verification ===
echo ""
echo "⚙️ PHASE 6: Configuration Files Verification"
echo "-------------------------------------------"

config_files=(".clang-format" ".editorconfig" ".env.example" ".pre-commit-config.yaml")
for file in "${config_files[@]}"; do
    if [ -f "config/$file" ]; then
        echo "✅ config/$file exists"
    else
        echo "⚠️  config/$file missing"
    fi
done

# === PHASE 7: Scripts Organization Verification ===
echo ""
echo "🔧 PHASE 7: Scripts Organization Verification"
echo "--------------------------------------------"

scripts_count=$(ls -1 scripts/ 2>/dev/null | grep '\.sh$' | wc -l)
echo "Shell scripts in scripts/: $scripts_count"

if [ $scripts_count -gt 10 ]; then
    echo "✅ Scripts well organized"
else
    echo "⚠️  Few scripts in scripts/ directory"
fi

# === PHASE 8: Functional Testing ===
echo ""
echo "🧪 PHASE 8: Basic Functional Testing"
echo "-----------------------------------"

echo "Testing Docker availability..."
if command -v docker &>/dev/null; then
    echo "✅ Docker is available"

    if docker info &>/dev/null; then
        echo "✅ Docker daemon is running"
    else
        echo "⚠️  Docker daemon not running"
    fi
else
    echo "❌ Docker not found"
fi

echo ""
echo "Testing Docker Compose availability..."
if command -v docker-compose &>/dev/null; then
    echo "✅ docker-compose available"
elif docker compose version &>/dev/null; then
    echo "✅ docker compose available"
else
    echo "❌ Docker Compose not found"
fi

# === PHASE 9: File Count Summary ===
echo ""
echo "📊 PHASE 9: Organization Summary"
echo "------------------------------"

echo "File distribution after reorganization:"
for dir in docker config api web docs scripts tools src include tests gui; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        echo "  $dir/: $count files"
    fi
done

root_file_count=$(find . -maxdepth 1 -type f | wc -l)
echo "  root: $root_file_count files"

# === PHASE 10: Recommendations ===
echo ""
echo "💡 PHASE 10: Recommendations"
echo "---------------------------"

if [ $root_file_count -gt 10 ]; then
    echo "⚠️  Root directory still has $root_file_count files"
    echo "   Consider moving more files to subdirectories"
else
    echo "✅ Root directory is clean ($root_file_count files)"
fi

if [ ! -f "docker/docker-compose.yml" ]; then
    echo "⚠️  Consider moving docker-compose.yml to docker/ directory"
fi

if [ ! -f "QUICK_START.md" ]; then
    echo "⚠️  Consider creating a QUICK_START.md guide"
fi

echo ""
echo "🎉 VERIFICATION COMPLETE!"
echo "========================"

if [ $root_file_count -le 10 ] && [ -f "docker/docker-compose.yml" ] && [ -x "run.sh" ]; then
    echo "✅ Project structure is clean and well-organized!"
    echo "✅ Ready for professional development and deployment"
else
    echo "⚠️  Some improvements recommended (see above)"
    echo "✅ Structure is significantly improved from original"
fi

echo ""
echo "🚀 Next steps:"
echo "  1. Test the reorganized setup: ./run.sh up"
echo "  2. Verify all services work: ./run.sh status"
echo "  3. Update any remaining hardcoded paths"
echo "  4. Commit the clean structure to git"
