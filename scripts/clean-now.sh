#!/bin/bash
set -e

cd /home/kevin/Projects/computer-vision

echo "🧹 IMMEDIATE PROJECT CLEANUP - SIMPLIFIED VERSION"
echo "================================================"
echo "Cleaning up the most obvious clutter right now!"
echo ""

# Create backup
BACKUP="../computer-vision-backup-$(date +%Y%m%d-%H%M%S)"
echo "📦 Creating backup: $BACKUP"
cp -r . "$BACKUP"

# Create directories that don't exist
echo "📁 Creating necessary directories..."
mkdir -p docker config api web deployment

# Move most obvious Docker files
echo "🐳 Moving Docker files to docker/..."
if [ -f "Dockerfile" ]; then
    mv Dockerfile docker/ && echo "  ✅ Dockerfile → docker/"
fi
if [ -f "docker-compose.yml" ]; then
    mv docker-compose.yml docker/ && echo "  ✅ docker-compose.yml → docker/"
fi
if [ -f "docker-compose.yml.backup" ]; then
    mv docker-compose.yml.backup docker/ && echo "  ✅ docker-compose.yml.backup → docker/"
fi
if [ -f "docker-compose.yml.new" ]; then
    mv docker-compose.yml.new docker/ && echo "  ✅ docker-compose.yml.new → docker/"
fi

# Move Docker documentation
for file in DOCKER_*.md; do
    if [ -f "$file" ]; then
        mv "$file" docker/ && echo "  ✅ $file → docker/"
    fi
done

# Move most scripts to scripts/ (keep run.sh)
echo "🔧 Moving scripts to scripts/..."
for script in *.sh; do
    if [ -f "$script" ] && [ "$script" != "run.sh" ]; then
        mv "$script" scripts/ && echo "  ✅ $script → scripts/"
    fi
done

# Move documentation files to docs/
echo "📚 Moving documentation to docs/..."
for doc in *.md; do
    if [ -f "$doc" ] && [ "$doc" != "README.md" ]; then
        mv "$doc" docs/ && echo "  ✅ $doc → docs/"
    fi
done

# Move configuration files
echo "⚙️ Moving configuration files to config/..."
if [ -f ".clang-format" ]; then
    mv .clang-format config/ && echo "  ✅ .clang-format → config/"
fi
if [ -f ".editorconfig" ]; then
    mv .editorconfig config/ && echo "  ✅ .editorconfig → config/"
fi
if [ -f ".env.example" ]; then
    mv .env.example config/ && echo "  ✅ .env.example → config/"
fi
if [ -f ".pre-commit-config.yaml" ]; then
    mv .pre-commit-config.yaml config/ && echo "  ✅ .pre-commit-config.yaml → config/"
fi

# Move API files
echo "🐍 Moving API files to api/..."
if [ -f "api-server.py" ]; then
    mv api-server.py api/ && echo "  ✅ api-server.py → api/"
fi

# Move web files
echo "🌐 Moving web files to web/..."
if [ -f "test-connection.html" ]; then
    mv test-connection.html web/ && echo "  ✅ test-connection.html → web/"
fi

# Move tools
echo "🛠️ Moving tools..."
if [ -f "Universal_Docker_Development_Strategy.ipynb" ]; then
    mv Universal_Docker_Development_Strategy.ipynb tools/ && echo "  ✅ Jupyter notebook → tools/"
fi

# Move test files
echo "🧪 Moving test files..."
if [ -f "test_args.cpp" ]; then
    mv test_args.cpp tests/ && echo "  ✅ test_args.cpp → tests/"
fi

# Create symlinks for important scripts in root if moved
if [ -f "scripts/run.sh" ] && [ ! -f "run.sh" ]; then
    ln -s scripts/run.sh run.sh && echo "  🔗 Created symlink: run.sh → scripts/run.sh"
fi

echo ""
echo "📊 CLEANUP RESULTS:"
echo "=================="
ROOT_FILES=$(ls -1 | wc -l)
echo "Files remaining in root: $ROOT_FILES"

echo ""
echo "📁 Organized directories:"
for dir in docker config api web docs scripts tools; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  $dir/: $count files"
    fi
done

echo ""
echo "🎉 IMMEDIATE CLEANUP COMPLETE!"
echo "============================="
echo ""
echo "✅ Root directory is much cleaner!"
echo "✅ Files are logically organized!"
echo "✅ Project structure is professional!"
echo ""
echo "💾 Backup saved to: $BACKUP"
echo "🚀 Ready to continue development with clean structure!"
