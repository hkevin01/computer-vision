#!/bin/bash
set -e

echo "🔧 Docker Service Debug & Restart"
echo "================================="

cd /home/kevin/Projects/computer-vision

# Check if services are currently running
echo "1. Current status check..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

echo "Current containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "Compose services:"
$COMPOSE_CMD ps

# Let's see what happened with the original startup
echo ""
echo "2. Checking for any running stereo services..."
STEREO_CONTAINERS=$(docker ps --filter "name=stereo" --format "{{.Names}}")
if [ ! -z "$STEREO_CONTAINERS" ]; then
    echo "Found running containers: $STEREO_CONTAINERS"
    for container in $STEREO_CONTAINERS; do
        echo "--- Logs for $container ---"
        docker logs --tail=10 "$container"
    done
else
    echo "No stereo containers currently running"
fi

# Check for stopped containers
echo ""
echo "3. Checking for stopped stereo containers..."
STOPPED_CONTAINERS=$(docker ps -a --filter "name=stereo" --filter "status=exited" --format "{{.Names}}")
if [ ! -z "$STOPPED_CONTAINERS" ]; then
    echo "Found stopped containers: $STOPPED_CONTAINERS"
    for container in $STOPPED_CONTAINERS; do
        echo "--- Last logs for $container ---"
        docker logs --tail=20 "$container"
    done
else
    echo "No stopped stereo containers found"
fi

# Test if we can directly reach localhost:3000 and localhost:8080
echo ""
echo "4. Direct connectivity test..."
echo "Testing port 3000:"
timeout 3 bash -c "</dev/tcp/localhost/3000" 2>/dev/null && echo "✅ Port 3000 accessible" || echo "❌ Port 3000 not accessible"

echo "Testing port 8080:"
timeout 3 bash -c "</dev/tcp/localhost/8080" 2>/dev/null && echo "✅ Port 8080 accessible" || echo "❌ Port 8080 not accessible"

# Check what services Docker thinks are running
echo ""
echo "5. Docker internal status..."
echo "Networks:"
docker network ls | grep stereo || echo "No stereo networks"

echo ""
echo "Volumes:"
docker volume ls | grep stereo || echo "No stereo volumes"

echo ""
echo "Images:"
docker images | grep -E "stereo|nginx" || echo "No stereo/nginx images"

# Try to restart if services are supposed to be running
echo ""
echo "6. Attempting restart if needed..."
echo "Use './run.sh down && ./run.sh up' to restart services"
echo "Or run this script's recommendations below:"

# Recommendations
echo ""
echo "🔍 DIAGNOSIS RECOMMENDATIONS:"
echo "If no containers are running:"
echo "  ./run.sh up"
echo ""
echo "If containers exited with errors:"
echo "  ./run.sh down && ./run.sh build && ./run.sh up"
echo ""
echo "If services are running but not responding:"
echo "  Check if ports are bound to different interfaces"
echo "  ./run.sh logs api"
echo "  ./run.sh logs gui"
