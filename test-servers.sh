#!/bin/bash

echo "🧪 Testing API Server Directly"
echo "============================="

cd /home/kevin/Projects/computer-vision

# Start the API server in background
echo "1. Starting temporary API server..."
python3 api-server.py &
API_PID=$!

# Wait for it to start
sleep 3

# Test the API
echo ""
echo "2. Testing API endpoints..."
echo "Health check:"
curl -s "http://localhost:8080/health" | python3 -m json.tool || echo "❌ Health check failed"

echo ""
echo "Status check:"
curl -s "http://localhost:8080/api/status" | python3 -m json.tool || echo "❌ Status check failed"

echo ""
echo "3. Testing GUI connectivity..."
if [ -f "gui/index.html" ]; then
    echo "Starting simple GUI server..."
    cd gui
    python3 -m http.server 3000 &
    GUI_PID=$!
    cd ..

    sleep 3

    echo "Testing GUI access:"
    curl -s -I "http://localhost:3000" | head -1 || echo "❌ GUI access failed"
else
    echo "❌ GUI files not found"
fi

echo ""
echo "4. Services are running. Test in browser:"
echo "  🌐 GUI: http://localhost:3000"
echo "  🔗 API: http://localhost:8080/health"

echo ""
echo "Press any key to stop servers..."
read -n 1

# Cleanup
echo ""
echo "5. Stopping servers..."
[ ! -z "$API_PID" ] && kill $API_PID 2>/dev/null
[ ! -z "$GUI_PID" ] && kill $GUI_PID 2>/dev/null

echo "✅ Test complete"
