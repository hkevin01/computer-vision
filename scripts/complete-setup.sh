#!/bin/bash
set -e

echo "🚀 FINAL Docker-First Stereo Vision Setup"
echo "========================================"

cd /home/kevin/Projects/computer-vision

# === PHASE 1: Environment Verification ===
echo ""
echo "📋 PHASE 1: Verifying Environment"
echo "--------------------------------"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon not running. Please start Docker."
    exit 1
fi

echo "✅ Docker is ready"

# Check Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

echo "✅ Docker Compose is ready ($COMPOSE_CMD)"

# === PHASE 2: File Structure Verification ===
echo ""
echo "📁 PHASE 2: Verifying File Structure"
echo "-----------------------------------"

# Check critical files
CRITICAL_FILES=(
    "Dockerfile"
    "docker-compose.yml"
    ".env"
    "run.sh"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Check GUI directory
if [ ! -d "gui" ]; then
    echo "⚠️  GUI directory missing, creating..."
    ./run.sh gui:create --force
fi

# Verify GUI files
GUI_FILES=("gui/index.html" "gui/styles.css" "gui/app.js" "gui/Dockerfile" "gui/config.json")
for file in "${GUI_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        echo "Creating GUI scaffold..."
        ./run.sh gui:create --force
        break
    fi
done

# === PHASE 3: Clean Slate ===
echo ""
echo "🧹 PHASE 3: Clean Environment"
echo "-----------------------------"

echo "Stopping any existing services..."
$COMPOSE_CMD down --remove-orphans --volumes || echo "No services to stop"

echo "Cleaning up orphaned containers..."
docker container prune -f

echo "✅ Environment cleaned"

# === PHASE 4: Build Services ===
echo ""
echo "🔨 PHASE 4: Building Services"
echo "-----------------------------"

echo "Building API service..."
$COMPOSE_CMD build --no-cache api

echo "Building GUI service..."
$COMPOSE_CMD build --no-cache gui

echo "✅ Services built successfully"

# === PHASE 5: Start Services ===
echo ""
echo "🌟 PHASE 5: Starting Services"
echo "-----------------------------"

echo "Starting API service first..."
$COMPOSE_CMD up -d api

echo "Waiting for API to be ready..."
for i in {1..12}; do
    if curl -s -f "http://localhost:8080/health" > /dev/null 2>&1; then
        echo "✅ API is healthy"
        break
    else
        echo "⏳ Waiting for API... ($i/12)"
        sleep 5
    fi
done

echo "Starting GUI service..."
$COMPOSE_CMD up -d gui

echo "Waiting for GUI to be ready..."
for i in {1..8}; do
    if curl -s -f "http://localhost:3000/health" > /dev/null 2>&1; then
        echo "✅ GUI is healthy"
        break
    else
        echo "⏳ Waiting for GUI... ($i/8)"
        sleep 5
    fi
done

# === PHASE 6: Verification ===
echo ""
echo "✅ PHASE 6: Final Verification"
echo "------------------------------"

echo "Container status:"
$COMPOSE_CMD ps

echo ""
echo "Testing services:"

# Test API
echo -n "API (http://localhost:8080/health): "
if curl -s -f "http://localhost:8080/health" > /dev/null; then
    echo "✅ Working"
else
    echo "❌ Not responding"
    echo "API logs:"
    $COMPOSE_CMD logs --tail=10 api
fi

# Test GUI
echo -n "GUI (http://localhost:3000): "
if curl -s -f "http://localhost:3000" > /dev/null; then
    echo "✅ Working"
else
    echo "❌ Not responding"
    echo "GUI logs:"
    $COMPOSE_CMD logs --tail=10 gui
fi

# === PHASE 7: Success Information ===
echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================="
echo ""
echo "📍 Your Docker-first Stereo Vision application is ready:"
echo ""
echo "🌐 Web GUI:     http://localhost:3000"
echo "🔗 API Server:  http://localhost:8080"
echo "📊 Health:      http://localhost:8080/health"
echo "📡 API Status:  http://localhost:8080/api/status"
echo ""
echo "🛠️  Management Commands:"
echo "  ./run.sh status          # Check service status"
echo "  ./run.sh logs api        # View API logs"
echo "  ./run.sh logs gui        # View GUI logs"
echo "  ./run.sh gui:open        # Open web interface"
echo "  ./run.sh down            # Stop all services"
echo "  ./run.sh up              # Start all services"
echo ""
echo "🔍 If something isn't working:"
echo "  1. Check logs: ./run.sh logs"
echo "  2. Restart: ./run.sh down && ./run.sh up"
echo "  3. Rebuild: ./run.sh build && ./run.sh up"
echo ""
echo "📖 See DOCKER_RUNNER_README.md for detailed documentation"
echo ""
echo "✨ Your Docker-first computer vision solution is live!"
