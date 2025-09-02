#!/bin/bash
set -e

MODE=${APP_MODE:-gui}

case "$MODE" in
    "api")
        echo "🚀 Starting in API mode..."
        python3 -c "
import http.server, socketserver, json

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
        elif '/api/' in self.path:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success', 'endpoint': self.path}).encode())
        else:
            super().do_GET()

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'success', 'method': 'POST'}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

with socketserver.TCPServer(('', 8080), Handler) as httpd:
    print('🌐 API Server running on port 8080')
    httpd.serve_forever()
"
        ;;
    "gui")
        echo "🎯 Starting in GUI mode..."
        export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
        export DISPLAY=${DISPLAY:-:0}
        exec ./build/stereo_vision_app
        ;;
    "simple")
        echo "🚀 Starting in simple mode..."
        export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
        exec ./build/stereo_vision_app_simple
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Available modes: api, gui, simple"
        exit 1
        ;;
esac
