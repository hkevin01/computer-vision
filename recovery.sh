#!/bin/bash
set -e

echo "🛠️ Docker Services Recovery and Restart"
echo "======================================"

cd /home/kevin/Projects/computer-vision

# 1. Stop everything cleanly
echo "1. Stopping all services..."
./run.sh down || echo "No services to stop"

# 2. Check GUI structure
echo ""
echo "2. Verifying GUI structure..."
echo "GUI files:"
ls -la gui/ || echo "No GUI directory"

# 3. Clean any orphaned containers
echo ""
echo "3. Cleaning orphaned containers..."
docker container prune -f

# 4. Check if we need to create GUI first
echo ""
echo "4. Ensuring GUI exists..."
if [ ! -f "gui/index.html" ] || [ ! -f "gui/Dockerfile" ]; then
    echo "⚠️  GUI files missing, creating..."
    ./run.sh gui:create --force
fi

# 5. Build API service first
echo ""
echo "5. Building API service..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

$COMPOSE_CMD build api --no-cache

# 6. Build GUI service
echo ""
echo "6. Building GUI service..."
$COMPOSE_CMD build gui --no-cache

# 7. Start API service first
echo ""
echo "7. Starting API service..."
$COMPOSE_CMD up -d api
sleep 15

# 8. Check API health
echo ""
echo "8. Checking API health..."
for i in {1..12}; do
    if curl -s "http://localhost:8080/health" > /dev/null; then
        echo "✅ API is healthy"
        break
    else
        echo "⏳ Waiting for API... ($i/12)"
        sleep 5
    fi
done

# 9. Start GUI service
echo ""
echo "9. Starting GUI service..."
$COMPOSE_CMD up -d gui
sleep 10

# 10. Check GUI health
echo ""
echo "10. Checking GUI health..."
for i in {1..6}; do
    if curl -s "http://localhost:3000/health" > /dev/null; then
        echo "✅ GUI is healthy"
        break
    else
        echo "⏳ Waiting for GUI... ($i/6)"
        sleep 5
    fi
done

# 11. Final status
echo ""
echo "11. Final service status:"
./run.sh status

# 12. Test connections
echo ""
echo "12. Connection tests:"
echo "GUI: http://localhost:3000"
curl -s -I "http://localhost:3000" | head -1 || echo "❌ GUI connection failed"

echo "API: http://localhost:8080"
curl -s -I "http://localhost:8080" | head -1 || echo "❌ API connection failed"

echo ""
echo "🎉 Recovery complete! Try:"
echo "  ./run.sh gui:open    # Open web interface"
echo "  ./run.sh logs api    # View API logs"
echo "  ./run.sh logs gui    # View GUI logs"
