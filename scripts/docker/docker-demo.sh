#!/usr/bin/env bash
# Interactive Docker demo (moved from scripts/legacy)
echo "This is the docker demo (new location: scripts/docker/docker-demo.sh)"
echo "Running demo steps..."

if [ -f "./docker-compose.yml" ]; then
    echo "Found docker-compose.yml — printing status"
    docker-compose ps
else
    echo "No docker-compose.yml found in project root."
fi

exit 0
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

echo "🚀 Ready to use Docker with your Stereo Vision Application!"
echo "   Next steps: Copy .env.example to .env and run 'docker compose build'"
