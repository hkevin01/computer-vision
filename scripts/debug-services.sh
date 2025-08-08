#!/bin/bash
cd /home/kevin/Projects/computer-vision
echo "=== Docker Container Status ==="
docker ps -a --filter "name=stereo"

echo ""
echo "=== Docker Compose Status ==="
docker-compose ps

echo ""
echo "=== Recent Container Logs ==="
echo "--- API Service ---"
docker-compose logs --tail=10 api

echo "--- GUI Service ---"
docker-compose logs --tail=10 gui

echo ""
echo "=== Network Status ==="
docker network ls | grep stereo

echo ""
echo "=== Port Check ==="
echo "API Port 8080:"
netstat -tlnp | grep :8080 || echo "Not bound"
echo "GUI Port 3000:"
netstat -tlnp | grep :3000 || echo "Not bound"
