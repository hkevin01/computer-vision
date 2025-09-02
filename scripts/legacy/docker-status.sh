#!/bin/bash

echo "🔍 Current Docker Status Check"
echo "============================="

# Check current docker status
echo "1. Currently running containers:"
docker ps --all --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"

echo ""
echo "2. Docker images available:"
docker images | grep -E "stereo|nginx" || echo "No stereo/nginx images found"

echo ""
echo "3. Docker networks:"
docker network ls | grep stereo || echo "No stereo networks found"

echo ""
echo "4. Quick port check:"
echo "Port 3000: $(ss -tulnp | grep :3000 | wc -l) listeners"
echo "Port 8080: $(ss -tulnp | grep :8080 | wc -l) listeners"

echo ""
echo "5. Docker compose status:"
cd /home/kevin/Projects/computer-vision
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

echo ""
echo "6. Last 10 lines of any container logs:"
CONTAINERS=$(docker ps -a --filter "name=stereo" --format "{{.Names}}")
if [ ! -z "$CONTAINERS" ]; then
    for container in $CONTAINERS; do
        echo "--- $container ---"
        docker logs --tail=5 "$container" 2>&1 || echo "No logs available"
    done
else
    echo "No stereo containers found"
fi
