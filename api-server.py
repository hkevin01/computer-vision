#!/usr/bin/env python3
"""
Temporary API server for Stereo Vision Application
This provides a working API endpoint until the C++ API is ready
"""

import http.server
import json
import socketserver
import urllib.parse
from datetime import datetime


class StereoVisionAPIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "Stereo Vision API",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "cameras": {
                    "left": {"connected": True, "status": "ready"},
                    "right": {"connected": True, "status": "ready"}
                },
                "processing": {
                    "calibrated": False,
                    "stereo_ready": True,
                    "last_processing": None
                },
                "system": {
                    "cpu_usage": "15%",
                    "memory_usage": "42%",
                    "uptime": "5m 32s"
                }
            }
            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == '/api/cameras':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "cameras": [
                    {
                        "id": 0,
                        "name": "Left Camera",
                        "connected": True,
                        "resolution": "1920x1080",
                        "fps": 30
                    },
                    {
                        "id": 1,
                        "name": "Right Camera",
                        "connected": True,
                        "resolution": "1920x1080",
                        "fps": 30
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"error": "Endpoint not found"}
            self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {
            "message": "Processing started",
            "timestamp": datetime.now().isoformat()
        }
        self.wfile.write(json.dumps(response).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == "__main__":
    PORT = 8080
    print(f"🌐 Starting Stereo Vision API Server on port {PORT}")
    print("📍 Endpoints available:")
    print("   GET  /health - Health check")
    print("   GET  /api/status - System status")
    print("   GET  /api/cameras - Camera information")
    print("   POST /api/* - Processing endpoints")

    with socketserver.TCPServer(("", PORT), StereoVisionAPIHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            httpd.shutdown()
            httpd.shutdown()
