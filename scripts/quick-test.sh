#!/bin/bash
# Simple test to check Docker services without building

echo "🧪 Quick Docker Service Test"
echo "=========================="

cd /home/kevin/Projects/computer-vision

echo "1. Checking what containers are currently running..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "2. Checking if stereo vision containers exist..."
docker ps -a --filter "label=com.docker.compose.project=computer-vision" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No compose containers found"

echo ""
echo "3. Checking compose services status..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

$COMPOSE_CMD ps 2>/dev/null || echo "No compose services found"

echo ""
echo "4. Testing direct network connectivity..."
echo "Testing GUI port 3000..."
timeout 3 bash -c "</dev/tcp/localhost/3000" 2>/dev/null && echo "✅ Port 3000 is open" || echo "❌ Port 3000 is not accessible"

echo "Testing API port 8080..."
timeout 3 bash -c "</dev/tcp/localhost/8080" 2>/dev/null && echo "✅ Port 8080 is open" || echo "❌ Port 8080 is not accessible"

echo ""
echo "5. Attempting to access services via HTTP..."
echo "GUI Health Check:"
curl -s --max-time 5 "http://localhost:3000/health" || echo "❌ GUI not responding"

echo ""
echo "API Health Check:"
curl -s --max-time 5 "http://localhost:8080/health" || echo "❌ API not responding"

echo ""
echo "6. Checking for running processes on these ports..."
netstat -tlnp 2>/dev/null | grep -E ':3000|:8080' || echo "No processes listening on GUI/API ports"
