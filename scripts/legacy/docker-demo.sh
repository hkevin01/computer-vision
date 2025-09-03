#!/bin/bash

# === Docker Setup Demonstration for Stereo Vision Application ===
# This script demonstrates the new Docker-first capabilities

set -e

echo "🚀 Docker Setup for Stereo Vision Application"
echo "=============================================="
echo ""

# Check Docker availability
echo "📋 Checking Docker Environment..."
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker not found. Please install Docker Desktop or Docker Engine."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker daemon not running. Please start Docker."
    exit 1
fi

echo "✅ Docker version: $(docker --version)"

if docker compose version >/dev/null 2>&1; then
    echo "✅ Docker Compose version: $(docker compose version --short)"
else
    echo "⚠️ Docker Compose not available. Some features will be limited."
fi

echo ""

# Show project structure
echo "📁 Project Structure:"
echo "   📄 Dockerfile - Multi-stage build configuration"
echo "   📄 docker-compose.yml - Service orchestration"
echo "   📄 .env.example - Environment template"
echo "   📄 run.sh - Enhanced script with Docker support"
echo ""

# Check files
echo "🔍 Checking Docker Files..."
for file in Dockerfile docker-compose.yml .env.example; do
    if [[ -f "$file" ]]; then
        echo "   ✅ $file ($(stat -c%s "$file") bytes)"
    else
        echo "   ❌ $file (missing)"
    fi
done
echo ""

# Show available commands
echo "🛠️ Available Docker Commands:"
echo ""
echo "   Basic Commands:"
echo "   docker compose build              # Build all images"
echo "   docker compose up -d              # Start application"
echo "   docker compose down               # Stop and remove containers"
echo "   docker compose logs -f            # View logs"
echo "   docker compose ps                 # Show running containers"
echo ""
echo "   Development Commands:"
echo "   docker compose --profile dev up   # Start development environment"
echo "   docker compose exec stereo-vision-app bash   # Open shell"
echo ""
echo "   Example Builds:"
echo "   ENABLE_CUDA=true docker compose build        # With NVIDIA GPU"
echo "   ENABLE_HIP=true docker compose build         # With AMD GPU"
echo ""

# Show environment setup
echo "⚙️ Environment Setup:"
echo "   1. Copy environment template:"
echo "      cp .env.example .env"
echo ""
#!/bin/bash
# Legacy shim to new docker demo
NEW="$(dirname "$0")/../docker/docker-demo.sh"
if [ -x "$NEW" ]; then
    exec "$NEW" "$@"
else
    echo "docker-demo.sh not found in new location: $NEW"
    echo "Falling back to legacy behavior."
    # legacy behavior placeholder
    exit 2
fi
