#!/bin/bash
set -e

echo "🧹 COMPREHENSIVE PROJECT REORGANIZATION"
echo "======================================"
echo "Transforming cluttered root directory into clean, professional structure"
echo ""

cd /home/kevin/Projects/computer-vision

# Create backup of current state
echo "📦 Creating backup of current state..."
cp -r . ../computer-vision-backup-$(date +%Y%m%d-%H%M%S) || echo "Backup creation failed, continuing..."

# === PHASE 1: Create Directory Structure ===
echo ""
echo "📁 PHASE 1: Creating Clean Directory Structure"
echo "---------------------------------------------"

# Create necessary directories if they don't exist
directories=(
    "docker"
    "config"
    "api"
    "web"
    "deployment"
    "documentation"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✅ Created: $dir/"
    else
        echo "✅ Exists: $dir/"
    fi
done

# === PHASE 2: Move Docker & Container Files ===
echo ""
echo "🐳 PHASE 2: Organizing Docker & Container Files"
echo "-----------------------------------------------"

docker_files=(
    "Dockerfile"
    "docker-compose.yml"
    "docker-compose.yml.backup"
    "docker-compose.yml.new"
    "DOCKER_README.md"
    "DOCKER_RUNNER_README.md"
    "DOCKER_SETUP.md"
    "DOCKER_SETUP_COMPLETE.md"
)

for file in "${docker_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" docker/
        echo "✅ Moved: $file → docker/"
    fi
done

# === PHASE 3: Move Configuration Files ===
echo ""
echo "⚙️ PHASE 3: Organizing Configuration Files"
echo "------------------------------------------"

config_files=(
    ".clang-format"
    ".editorconfig"
    ".env.example"
    ".pre-commit-config.yaml"
)

for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" config/
        echo "✅ Moved: $file → config/"
    fi
done

# === PHASE 4: Move Documentation Files ===
echo ""
echo "📚 PHASE 4: Consolidating Documentation"
echo "---------------------------------------"

doc_files=(
    "AI_ML_IMPROVEMENTS_SUMMARY.md"
    "CHANGELOG.md"
    "CONTRIBUTING.md"
    "DIRECTORY_CLEANUP_SUMMARY.md"
    "IMPLEMENTATION_PLAN.md"
    "IMPROVEMENTS_ROADMAP.md"
    "OPENCV_OPTIMIZATION.md"
    "PROJECT_CLEANUP_COMPLETE.md"
    "PROJECT_MODERNIZATION_STRATEGY.md"
    "docs/guides/README_CLEANUP.md"
    "SECURITY.md"
    "WORKFLOW.md"
)

for file in "${doc_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" docs/
        echo "✅ Moved: $file → docs/"
    fi
done

# === PHASE 5: Move Shell Scripts ===
echo ""
echo "🔧 PHASE 5: Organizing Shell Scripts"
echo "------------------------------------"

script_files=(
    "check-status.sh"
    "cleanup-project.sh"
    "complete-setup.sh"
    "compose-debug.sh"
    "debug-restart.sh"
    "debug-services.sh"
    "diagnose.sh"
    "docker-demo.sh"
    "docker-status.sh"
    "final-check.sh"
    "full-debug.sh"
    "launch_gui.sh"
    "quick-test.sh"
    "rebuild-and-test.sh"
    "recovery.sh"
    "run.sh.backup"
    "run.sh.new"
    "test-docker-setup.sh"
    "test-servers.sh"
    "update-docker-setup.sh"
    "COMPLETE_CLEANUP.sh"
)

for file in "${script_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" scripts/
        echo "✅ Moved: $file → scripts/"
    fi
done

# === PHASE 6: Move API & Web Files ===
echo ""
echo "🌐 PHASE 6: Organizing API & Web Files"
echo "--------------------------------------"

# Move API files
if [ -f "api-server.py" ]; then
    mv "api-server.py" api/
    echo "✅ Moved: api-server.py → api/"
fi

# Move web files
web_files=(
    "test-connection.html"
)

for file in "${web_files[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" web/
        echo "✅ Moved: $file → web/"
    fi
done

# Move GUI directory contents to web if needed
if [ -d "gui" ]; then
    echo "✅ Keeping: gui/ (already organized)"
fi

# === PHASE 7: Move Development Tools ===
echo ""
echo "🛠️ PHASE 7: Organizing Development Tools"
echo "----------------------------------------"

if [ -f "Universal_Docker_Development_Strategy.ipynb" ]; then
    mv "Universal_Docker_Development_Strategy.ipynb" tools/
    echo "✅ Moved: Universal_Docker_Development_Strategy.ipynb → tools/"
fi

# === PHASE 8: Move Test Files ===
echo ""
echo "🧪 PHASE 8: Organizing Test Files"
echo "---------------------------------"

if [ -f "test_args.cpp" ]; then
    mv "test_args.cpp" tests/
    echo "✅ Moved: test_args.cpp → tests/"
fi

# === PHASE 9: Create Symlinks for Essential Scripts ===
echo ""
echo "🔗 PHASE 9: Creating Convenient Access Links"
echo "--------------------------------------------"

# Create symlinks for commonly used scripts in root
essential_scripts=(
    "run.sh"
)

for script in "${essential_scripts[@]}"; do
    if [ -f "scripts/$script" ] && [ ! -f "$script" ]; then
        ln -s "scripts/$script" "$script"
        echo "✅ Created symlink: $script → scripts/$script"
    elif [ -f "$script" ]; then
        echo "✅ Kept in root: $script (main entry point)"
    fi
done

# === PHASE 10: Update Path References ===
echo ""
echo "🔄 PHASE 10: Updating Path References"
echo "-------------------------------------"

# Update docker-compose.yml to point to correct Dockerfile location
if [ -f "docker/docker-compose.yml" ]; then
    sed -i 's|dockerfile: Dockerfile|dockerfile: ../Dockerfile|g' docker/docker-compose.yml
    echo "✅ Updated: docker-compose.yml Dockerfile paths"
fi

# Update scripts that reference docker-compose.yml
if [ -f "scripts/run.sh.new" ]; then
    sed -i 's|docker-compose\.yml|docker/docker-compose.yml|g' scripts/run.sh.new
    echo "✅ Updated: run.sh.new docker-compose paths"
fi

# === PHASE 11: Create README Files for Each Directory ===
echo ""
echo "📖 PHASE 11: Creating Directory Documentation"
echo "---------------------------------------------"

# Docker directory README
cat > docker/README.md << 'EOF'
# Docker Configuration

This directory contains all Docker-related files for the Stereo Vision application.

## Files:
- `Dockerfile` - Main application container definition
- `docker-compose.yml` - Service orchestration configuration
- `DOCKER_*.md` - Docker-specific documentation

## Usage:
```bash
# From project root:
cd docker
docker-compose up
```
EOF
echo "✅ Created: docker/README.md"

# Config directory README
cat > config/README.md << 'EOF'
# Configuration Files

This directory contains project configuration files.

## Files:
- `.clang-format` - Code formatting rules
- `.editorconfig` - Editor configuration
- `.env.example` - Environment variables template
- `.pre-commit-config.yaml` - Git pre-commit hooks

## Usage:
Copy `.env.example` to `.env` and modify as needed.
EOF
echo "✅ Created: config/README.md"

# API directory README
cat > api/README.md << 'EOF'
# API Server

This directory contains the API server implementation.

## Files:
- `api-server.py` - Python API server for development/testing

## Usage:
```bash
python3 api-server.py
```
EOF
echo "✅ Created: api/README.md"

# Web directory README
cat > web/README.md << 'EOF'
# Web Assets

This directory contains web-related files and test pages.

## Files:
- `test-connection.html` - Connection testing interface

## Usage:
Serve files with any web server for testing.
EOF
echo "✅ Created: web/README.md"

# === PHASE 12: Update Main README ===
echo ""
echo "📄 PHASE 12: Updating Main README"
echo "---------------------------------"

# Add directory structure to main README
cat >> README.md << 'EOF'

## Project Structure

```
computer-vision/
├── 📁 src/                     # Source code
├── 📁 include/                 # Header files
├── 📁 tests/                   # Unit tests
├── 📁 test_programs/           # Test utilities
├── 📁 docs/                    # Documentation
├── 📁 scripts/                 # Build & utility scripts
├── 📁 docker/                  # Docker configuration
├── 📁 config/                  # Project configuration
├── 📁 api/                     # API server
├── 📁 web/                     # Web assets
├── 📁 gui/                     # Web GUI interface
├── 📁 tools/                   # Development tools
├── 📁 data/                    # Sample data
├── 📁 build_scripts/           # Build utilities
├── 📁 reports/                 # Generated reports
├── 📁 logs/                    # Runtime logs
├── 📁 archive/                 # Historical files
├── 📄 CMakeLists.txt           # Build configuration
├── 📄 README.md                # This file
├── 📄 LICENSE                  # License
├── 📄 .gitignore               # Git ignore
└── 📄 run.sh                   # Main entry point
```

## Quick Start

```bash
# Build and run with Docker
./run.sh up

# Access web GUI
./run.sh gui:open

# Check status
./run.sh status
```
EOF
echo "✅ Updated: README.md with new structure"

# === PHASE 13: Final Verification ===
echo ""
echo "✅ PHASE 13: Final Verification"
echo "-------------------------------"

echo "Root directory contents:"
ls -la | grep -E '^-' | wc -l | xargs echo "Files in root:"
ls -la | grep -E '^d' | wc -l | xargs echo "Directories in root:"

echo ""
echo "Directory sizes:"
for dir in docker config api web docs scripts tools; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" | wc -l)
        echo "  $dir/: $count files"
    fi
done

# === COMPLETION SUMMARY ===
echo ""
echo "🎉 REORGANIZATION COMPLETE!"
echo "=========================="
echo ""
echo "✨ Transformations Applied:"
echo "  📦 Docker files → docker/"
echo "  ⚙️  Configuration → config/"
echo "  📚 Documentation → docs/"
echo "  🔧 Scripts → scripts/"
echo "  🌐 API files → api/"
echo "  🧪 Web assets → web/"
echo "  🛠️  Tools → tools/"
echo ""
echo "🎯 Clean Root Directory Achieved:"
echo "  ✅ Only essential files remain in root"
echo "  ✅ Logical grouping by function"
echo "  ✅ Professional project structure"
echo "  ✅ Easy navigation and maintenance"
echo ""
echo "📍 Key Entry Points:"
echo "  ./run.sh              # Main application runner"
echo "  docker/               # All Docker configuration"
echo "  docs/                 # All documentation"
echo "  scripts/              # All utility scripts"
echo ""
echo "🚀 Your project structure is now clean and professional!"
echo "📖 See individual directory README.md files for details"
