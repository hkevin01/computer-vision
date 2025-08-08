#!/bin/bash
set -e

echo "🔧 Rebuilding Docker-first Stereo Vision Services"
echo "=============================================="

cd /home/kevin/Projects/computer-vision

# Stop any running services first
echo "1. Stopping existing services..."
./run.sh down || echo "No services to stop"

# Check if GUI directory has all required files
echo ""
echo "2. Checking GUI structure..."
ls -la gui/
echo ""

# Build services with verbose output
echo "3. Building services..."
./run.sh build --verbose

# Start services
echo ""
echo "4. Starting services..."
./run.sh up

# Wait a moment and check status
echo ""
echo "5. Checking service status (after 10 second delay)..."
sleep 10
./run.sh status

echo ""
echo "6. Testing connections..."
echo "GUI: http://localhost:3000"
curl -s -I "http://localhost:3000" || echo "GUI not accessible"
echo ""
echo "API: http://localhost:8080"
curl -s -I "http://localhost:8080" || echo "API not accessible"
