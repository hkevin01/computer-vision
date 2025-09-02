#!/bin/bash

echo "🔍 Diagnosing Docker Services Status"
echo "=================================="

cd /home/kevin/Projects/computer-vision

# Check if Docker is running
echo "1. Docker daemon status:"
docker info > /dev/null 2>&1 && echo "✅ Docker is running" || echo "❌ Docker is not running"

# Check running containers
echo ""
echo "2. Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check all containers (including stopped)
echo ""
echo "3. All project containers:"
docker ps -a --filter "name=stereo" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check docker-compose services
echo ""
echo "4. Docker Compose services:"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

# Check specific ports
echo ""
echo "5. Port availability:"
echo "Port 3000 (GUI):"
ss -tulnp | grep :3000 || echo "  Not listening"
echo "Port 8080 (API):"
ss -tulnp | grep :8080 || echo "  Not listening"

# Check if we can reach the services
echo ""
echo "6. Service connectivity:"
echo "GUI Health Check:"
curl -s -w "%{http_code}" "http://localhost:3000/health" || echo "Connection failed"
echo ""
echo "API Health Check:"
curl -s -w "%{http_code}" "http://localhost:8080/health" || echo "Connection failed"

# Recent logs if any issues
echo ""
echo "7. Recent container logs:"
if docker ps --filter "name=stereo" | grep -q stereo; then
    echo "--- API logs (last 5 lines) ---"
    docker logs --tail=5 $(docker ps --filter "name=stereo.*api" --format "{{.Names}}" | head -1) 2>/dev/null || echo "No API container logs"
    echo "--- GUI logs (last 5 lines) ---"
    docker logs --tail=5 $(docker ps --filter "name=stereo.*gui" --format "{{.Names}}" | head -1) 2>/dev/null || echo "No GUI container logs"
else
    echo "No stereo containers running"
fi
