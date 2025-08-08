#!/bin/bash
cd /home/kevin/Projects/computer-vision

echo "📊 Running built-in status check..."
./run.sh status

echo ""
echo "📋 Running logs check for API..."
timeout 10 ./run.sh logs api || echo "No API logs or timeout"

echo ""
echo "📋 Running logs check for GUI..."
timeout 10 ./run.sh logs gui || echo "No GUI logs or timeout"
