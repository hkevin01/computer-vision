#!/bin/bash

echo "⚡ Direct Docker Compose Investigation"
echo "===================================="

cd /home/kevin/Projects/computer-vision

# Determine compose command
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "Using: docker-compose"
else
    COMPOSE_CMD="docker compose"
    echo "Using: docker compose"
fi

echo ""
echo "1. Current compose services status:"
$COMPOSE_CMD ps

echo ""
echo "2. Compose service configuration:"
$COMPOSE_CMD config --services

echo ""
echo "3. Checking if services are actually up:"
$COMPOSE_CMD ps --services | while read service; do
    echo "Service: $service"
    $COMPOSE_CMD ps "$service"
done

echo ""
echo "4. Recent logs for all services:"
$COMPOSE_CMD logs --tail=10

echo ""
echo "5. Attempting to restart all services:"
echo "Stopping services..."
$COMPOSE_CMD down

echo ""
echo "Building services fresh..."
$COMPOSE_CMD build --no-cache --parallel

echo ""
echo "Starting services with verbose output..."
$COMPOSE_CMD up -d

echo ""
echo "Waiting 15 seconds for startup..."
sleep 15

echo ""
echo "6. Post-restart status:"
$COMPOSE_CMD ps

echo ""
echo "7. Testing connectivity:"
echo "API Health Check:"
curl -s -m 5 "http://localhost:8080/health" || echo "❌ API not responding"

echo ""
echo "GUI Health Check:"
curl -s -m 5 "http://localhost:3000/health" || echo "❌ GUI not responding"

echo ""
echo "8. If still not working, check logs:"
echo "   ./run.sh logs api"
echo "   ./run.sh logs gui"
echo "   docker-compose logs"
