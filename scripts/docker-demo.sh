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
echo "   2. Edit .env file to configure:"
echo "      - IMAGE_NAME=stereo-vision:local"
echo "      - ENABLE_CUDA=true (for NVIDIA GPU)"
echo "      - ENABLE_HIP=true (for AMD GPU)"
echo "      - PORTS=8080:8080,8081:8081"
echo ""

# Show services
echo "🐳 Available Services:"
echo "   • stereo-vision-app    - Production application"
echo "   • stereo-vision-dev    - Development environment"
echo "   • stereo-vision-simple - Lightweight version"
echo ""

# Show quick start
echo "🏃 Quick Start Guide:"
echo "   1. cp .env.example .env"
echo "   2. docker compose build"
echo "   3. docker compose up -d"
echo "   4. docker compose logs -f"
echo ""

# Show enhanced run.sh
echo "🔧 Enhanced run.sh (when fully implemented):"
echo "   ./run.sh build      # Build Docker images"
echo "   ./run.sh up         # Start application"
echo "   ./run.sh dev        # Development mode"
echo "   ./run.sh shell      # Interactive shell"
echo "   ./run.sh status     # Show status"
echo ""

# Show legacy support
echo "🔄 Legacy Native Build Support:"
echo "   The original run.sh commands still work:"
echo "   ./run.sh --build-only    # Native CMake build"
echo "   ./run.sh --status        # Native build status"
echo "   ./run.sh --simple        # Simple application"
echo ""

echo "✨ Benefits of Docker Approach:"
echo "   ✅ Consistent environment across all systems"
echo "   ✅ No dependency conflicts with host system"
echo "   ✅ Easy setup for new developers"
echo "   ✅ Optional GPU acceleration (CUDA/HIP)"
echo "   ✅ Multiple build variants (production/dev/simple)"
echo "   ✅ Isolated GUI support with X11"
echo ""

echo "🚀 Ready to use Docker with your Stereo Vision Application!"
echo "   Next steps: Copy .env.example to .env and run 'docker compose build'"
