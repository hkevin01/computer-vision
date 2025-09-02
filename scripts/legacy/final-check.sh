#!/bin/bash

# Make all our debug and setup scripts executable
chmod +x /home/kevin/Projects/computer-vision/*.sh

echo "🔧 Made all scripts executable"

cd /home/kevin/Projects/computer-vision

echo ""
echo "📊 Current Service Status Check"
echo "==============================="

# Quick status check
echo "1. Docker containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "2. Docker Compose services:"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

echo ""
echo "3. Port connectivity:"
echo -n "Port 3000 (GUI): "
timeout 2 bash -c "</dev/tcp/localhost/3000" 2>/dev/null && echo "✅ Open" || echo "❌ Closed"

echo -n "Port 8080 (API): "
timeout 2 bash -c "</dev/tcp/localhost/8080" 2>/dev/null && echo "✅ Open" || echo "❌ Closed"

echo ""
echo "4. HTTP connectivity:"
echo -n "GUI Response: "
curl -s -I -m 3 "http://localhost:3000" | head -1 || echo "❌ No response"

echo -n "API Response: "
curl -s -I -m 3 "http://localhost:8080/health" | head -1 || echo "❌ No response"

echo ""
echo "🎯 READY TO PROCEED:"
echo ""
echo "If services are NOT running (most likely case):"
echo "  ./complete-setup.sh      # Complete automated setup"
echo ""
echo "If services ARE running but not accessible:"
echo "  ./debug-restart.sh       # Debug and restart"
echo ""
echo "For manual control:"
echo "  ./run.sh down            # Stop services"
echo "  ./run.sh up              # Start services"
echo "  ./run.sh status          # Check status"
echo ""
echo "✨ Ready to deploy your Docker-first stereo vision solution!"
