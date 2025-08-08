#!/bin/bash

echo "🔍 Comprehensive Docker Services Debug"
echo "====================================="

cd /home/kevin/Projects/computer-vision

# 1. Check what containers actually exist
echo "1. Docker container status:"
echo "   All containers:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"

echo ""
echo "   Stereo-related containers specifically:"
docker ps -a --filter "name=stereo" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}" || echo "   No stereo containers found"

# 2. Check compose services
echo ""
echo "2. Docker Compose status:"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

# 3. Check for logs if containers exist
echo ""
echo "3. Container logs (last 20 lines each):"
CONTAINERS=$(docker ps -a --filter "name=stereo" --format "{{.Names}}")
if [ ! -z "$CONTAINERS" ]; then
    for container in $CONTAINERS; do
        echo "   --- $container logs ---"
        docker logs --tail=20 "$container" 2>&1
        echo ""
    done
else
    echo "   No stereo containers to check logs for"
fi

# 4. Check networks
echo ""
echo "4. Network status:"
docker network ls | grep stereo || echo "   No stereo networks found"

# 5. Check images
echo ""
echo "5. Available images:"
docker images | grep -E "stereo|nginx" || echo "   No stereo/nginx images built"

# 6. Check if ports are occupied
echo ""
echo "6. Port status:"
netstat -tlnp 2>/dev/null | grep -E ':3000|:8080' || echo "   No processes on ports 3000 or 8080"

# 7. Try to manually start services if they're not running
echo ""
echo "7. Manual service check:"
if [ -z "$CONTAINERS" ]; then
    echo "   No stereo containers found. Trying to start services..."
    ./run.sh up || echo "   Failed to start services"

    # Wait and check again
    sleep 10
    echo "   Post-startup container check:"
    docker ps --filter "name=stereo" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "   Containers exist. Checking if they should be running..."
    RUNNING=$(docker ps --filter "name=stereo" --format "{{.Names}}")
    if [ -z "$RUNNING" ]; then
        echo "   Containers exist but are not running. Checking exit codes..."
        for container in $CONTAINERS; do
            EXIT_CODE=$(docker inspect "$container" --format='{{.State.ExitCode}}')
            echo "   $container exit code: $EXIT_CODE"
        done
    else
        echo "   Containers are running: $RUNNING"
    fi
fi

# 8. Final recommendations
echo ""
echo "8. 📋 RECOMMENDATIONS:"
if [ -z "$(docker ps --filter 'name=stereo' --format '{{.Names}}')" ]; then
    echo "   ❌ No stereo containers running"
    echo "   Try: ./run.sh down && ./run.sh build && ./run.sh up"
else
    echo "   ✅ Stereo containers are running"
    echo "   If services not accessible, check:"
    echo "   - ./run.sh logs api"
    echo "   - ./run.sh logs gui"
    echo "   - Network configuration in docker-compose.yml"
fi
