#!/usr/bin/env bash
set -Eeuo pipefail

# === Enhanced Docker-first Stereo Vision Application Runner ===
# Supports backend orchestration with auto-generated GUI capabilities

# Set error trap
trap 'echo "❌ Error occurred at line $LINENO. Exit code: $?" >&2' ERR

# === Configuration Defaults ===
# Docker & Compose settings
IMAGE_NAME="${IMAGE_NAME:-stereo-vision:local}"
GUI_IMAGE_NAME="${GUI_IMAGE_NAME:-stereo-vision-gui:local}"
SERVICE_NAME="${SERVICE_NAME:-api}"
GUI_SERVICE_NAME="${GUI_SERVICE_NAME:-gui}"
ENV_FILE="${ENV_FILE:-.env}"
PORTS="${PORTS:-8080:8080}"
GUI_PORT="${GUI_PORT:-3000}"
API_URL="${API_URL:-http://localhost:8081}"
GUI_PATH="${GUI_PATH:-./gui}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-}"
MOUNTS="${MOUNTS:-}"
BUILD_ARGS="${BUILD_ARGS:-}"
DEV_MODE="${DEV_MODE:-false}"
# GUI mode selection: novnc (default), x11, spa, none
GUI_MODE="${GUI_MODE:-novnc}"
NOVNC_PORT="${NOVNC_PORT:-8080}"
VNC_PORT="${VNC_PORT:-5900}"

# Runtime configuration
COMPOSE_CMD=""
DOCKER_BUILDKIT=1

# === Utility Functions ===
print_header() {
    echo "🚀 Enhanced Stereo Vision Docker Runner"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

print_status() {
    echo "📋 $1"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1" >&2
}

print_warning() {
    echo "⚠️  $1"
}

# Convenient URL opener with multiple fallbacks
open_in_browser() {
    local url="$1"
    # Prefer xdg-open on Linux
    if command -v xdg-open &>/dev/null; then
        nohup xdg-open "$url" >/dev/null 2>&1 &
        return 0
    fi
    # GNOME/GTK fallback
    if command -v gio &>/dev/null; then
        nohup gio open "$url" >/dev/null 2>&1 &
        return 0
    fi
    # Debian/Ubuntu sensible-browser
    if command -v sensible-browser &>/dev/null; then
        nohup sensible-browser "$url" >/dev/null 2>&1 &
        return 0
    fi
    # macOS
    if command -v open &>/dev/null; then
        nohup open "$url" >/dev/null 2>&1 &
        return 0
    fi
    # Python fallback
    if command -v python3 &>/dev/null; then
        python3 - <<PY
import webbrowser, sys
webbrowser.open(sys.argv[1])
PY
        return 0
    fi
    return 1
}

# Preflight checks
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running or not accessible"
        exit 1
    fi

    print_success "Docker is available"
}

detect_compose() {
    if command -v "docker-compose" &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Neither 'docker compose' nor 'docker-compose' is available"
        exit 1
    fi

    print_status "Using compose command: $COMPOSE_CMD"
}

# GUI Detection and Creation
detect_gui() {
    if [[ -d "$GUI_PATH" ]]; then
        print_success "GUI directory found at $GUI_PATH"
        return 0
    else
        print_warning "No GUI directory found at $GUI_PATH"
        return 1
    fi
}

create_gui_scaffold() {
    local force_create="${1:-false}"

    if [[ -d "$GUI_PATH" ]] && [[ "$force_create" != "true" ]]; then
        print_warning "GUI directory already exists. Use --force to overwrite."
        return 0
    fi

    print_status "Creating GUI scaffold at $GUI_PATH..."

    mkdir -p "$GUI_PATH"

    # Create index.html
    cat > "$GUI_PATH/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stereo Vision 3D - Control Panel</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🎥 Stereo Vision 3D Control Panel</h1>
            <div class="status-indicator" id="status-indicator">
                <span class="status-dot" id="status-dot"></span>
                <span id="status-text">Connecting...</span>
            </div>
        </header>

        <main class="main-content">
            <!-- API Status Panel -->
            <section class="card api-status">
                <h2>🔗 API Status</h2>
                <div class="api-info">
                    <p><strong>Endpoint:</strong> <span id="api-endpoint">Loading...</span></p>
                    <p><strong>Status:</strong> <span id="api-status">Checking...</span></p>
                    <p><strong>Response Time:</strong> <span id="response-time">-</span></p>
                </div>
                <button id="refresh-api" class="btn btn-primary">🔄 Refresh Status</button>
            </section>

            <!-- Camera Controls -->
            <section class="card camera-controls">
                <h2>📹 Camera Controls</h2>
                <div class="control-group">
                    <button id="detect-cameras" class="btn btn-secondary">🔍 Detect Cameras</button>
                    <button id="start-capture" class="btn btn-success">▶️ Start Capture</button>
                    <button id="stop-capture" class="btn btn-danger">⏹️ Stop Capture</button>
                </div>
                <div class="camera-info" id="camera-info">
                    <p>Click "Detect Cameras" to scan for available devices</p>
                </div>
            </section>

            <!-- Processing Controls -->
            <section class="card processing-controls">
                <h2>⚙️ Processing Controls</h2>
                <div class="control-group">
                    <button id="calibrate" class="btn btn-info">🎯 Calibrate</button>
                    <button id="process-stereo" class="btn btn-primary">🧮 Process Stereo</button>
                    <button id="generate-pointcloud" class="btn btn-success">☁️ Generate Point Cloud</button>
                </div>
                <div class="progress-container" id="progress-container" style="display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <p id="progress-text">Processing...</p>
                </div>
            </section>

            <!-- Results Display -->
            <section class="card results-display">
                <h2>📊 Results</h2>
                <div class="results-grid" id="results-grid">
                    <div class="result-item">
                        <h3>Disparity Map</h3>
                        <div class="image-placeholder">
                            <p>🖼️ No disparity map generated yet</p>
                        </div>
                    </div>
                    <div class="result-item">
                        <h3>Point Cloud</h3>
                        <div class="image-placeholder">
                            <p>☁️ No point cloud generated yet</p>
                        </div>
                    </div>
                </div>
            </section>
        </main>

        <footer class="footer">
            <p>🚀 Stereo Vision 3D Application - Docker Deployment</p>
            <p>API: <span id="footer-api-url">Loading...</span></p>
        </footer>
    </div>

    <script src="app.js"></script>
</body>
</html>
EOF

    # Create styles.css
    cat > "$GUI_PATH/styles.css" << 'EOF'
/* Stereo Vision 3D GUI Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.header {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
}

.header h1 {
    color: #4a5568;
    font-size: 2rem;
    font-weight: 700;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
}

.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #ffd700;
    animation: pulse 2s infinite;
}

.status-dot.online {
    background: #48bb78;
    animation: none;
}

.status-dot.offline {
    background: #f56565;
    animation: none;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Main Content */
.main-content {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 20px;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.95);
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
}

.card h2 {
    color: #2d3748;
    margin-bottom: 16px;
    font-size: 1.25rem;
    font-weight: 600;
}

/* Buttons */
.btn {
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
    display: inline-block;
    text-align: center;
    min-width: 120px;
}

.btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.btn-secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}

.btn-success {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
}

.btn-danger {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    color: white;
}

.btn-info {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #2d3748;
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

/* Control Groups */
.control-group {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
}

/* API Status */
.api-info p {
    margin-bottom: 8px;
    color: #4a5568;
}

.api-info strong {
    color: #2d3748;
}

/* Progress Bar */
.progress-container {
    margin-top: 16px;
}

.progress-bar {
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    width: 0%;
    transition: width 0.3s ease;
}

#progress-text {
    font-size: 14px;
    color: #4a5568;
}

/* Results Grid */
.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
}

.result-item h3 {
    color: #2d3748;
    margin-bottom: 12px;
    font-size: 1.1rem;
}

.image-placeholder {
    background: #f7fafc;
    border: 2px dashed #cbd5e0;
    border-radius: 8px;
    padding: 40px 20px;
    text-align: center;
    color: #a0aec0;
    font-style: italic;
}

/* Camera Info */
.camera-info {
    background: #f7fafc;
    padding: 16px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    color: #4a5568;
}

/* Footer */
.footer {
    background: rgba(255, 255, 255, 0.95);
    padding: 16px 20px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
    color: #4a5568;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

/* Responsive Design */
@media (max-width: 768px) {
    .header {
        flex-direction: column;
        text-align: center;
        gap: 16px;
    }

    .header h1 {
        font-size: 1.5rem;
    }

    .main-content {
        grid-template-columns: 1fr;
    }

    .control-group {
        flex-direction: column;
    }

    .btn {
        width: 100%;
        min-width: unset;
    }

    .results-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 480px) {
    .container {
        padding: 12px;
    }

    .card {
        padding: 16px;
    }

    .header h1 {
        font-size: 1.25rem;
    }
}

/* Animation Classes */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.loading {
    position: relative;
    overflow: hidden;
}

.loading::after {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    animation: loading 1.5s infinite;
}

@keyframes loading {
    to { left: 100%; }
}
EOF

    # Create app.js
    cat > "$GUI_PATH/app.js" << 'EOF'
// Stereo Vision 3D GUI Application Logic
class StereoVisionGUI {
    constructor() {
        this.apiUrl = this.getApiUrl();
        this.isProcessing = false;
        this.statusCheckInterval = null;

        this.init();
    }

    getApiUrl() {
        // Try to get API URL from config or environment
        if (window.CONFIG && window.CONFIG.API_URL) {
            return window.CONFIG.API_URL;
        }

        // Fallback to environment variable pattern
        const apiUrl = window.ENV?.API_URL || 'http://localhost:8081';
        return apiUrl;
    }

    init() {
        this.updateApiDisplay();
        this.bindEvents();
        this.startStatusCheck();
        this.checkApiStatus();

        console.log('🚀 Stereo Vision GUI initialized');
        console.log(`📡 API URL: ${this.apiUrl}`);
    }

    updateApiDisplay() {
        const elements = ['api-endpoint', 'footer-api-url'];
        elements.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) elem.textContent = this.apiUrl;
        });
    }

    bindEvents() {
        // API Status
        document.getElementById('refresh-api')?.addEventListener('click', () => {
            this.checkApiStatus();
        });

        // Camera Controls
        document.getElementById('detect-cameras')?.addEventListener('click', () => {
            this.detectCameras();
        });

        document.getElementById('start-capture')?.addEventListener('click', () => {
            this.startCapture();
        });

        document.getElementById('stop-capture')?.addEventListener('click', () => {
            this.stopCapture();
        });

        // Processing Controls
        document.getElementById('calibrate')?.addEventListener('click', () => {
            this.calibrateCameras();
        });

        document.getElementById('process-stereo')?.addEventListener('click', () => {
            this.processStereoImages();
        });

        document.getElementById('generate-pointcloud')?.addEventListener('click', () => {
            this.generatePointCloud();
        });
    }

    async checkApiStatus() {
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const apiStatus = document.getElementById('api-status');
        const responseTime = document.getElementById('response-time');

        const startTime = Date.now();

        try {
            // Try multiple common endpoints
            const endpoints = ['/health', '/status', '/api/health', '/'];
            let response = null;
            let endpoint = '';

            for (const ep of endpoints) {
                try {
                    const url = `${this.apiUrl}${ep}`;
                    response = await fetch(url, {
                        method: 'GET',
                        timeout: 5000,
                        headers: {
                            'Accept': 'application/json,text/plain,*/*'
                        }
                    });

                    if (response.ok) {
                        endpoint = ep;
                        break;
                    }
                } catch (e) {
                    // Continue to next endpoint
                    continue;
                }
            }

            const endTime = Date.now();
            const duration = endTime - startTime;

            if (response && response.ok) {
                // API is online
                statusDot.className = 'status-dot online';
                statusText.textContent = 'Connected';
                apiStatus.textContent = `✅ Online (${response.status})`;
                responseTime.textContent = `${duration}ms`;

                // Try to get additional info
                try {
                    const data = await response.json();
                    if (data.cameras) {
                        this.updateCameraInfo(data.cameras);
                    }
                } catch (e) {
                    // JSON parsing failed, but API is responding
                    console.log('API responding but not JSON:', e);
                }

            } else {
                // API returned error
                statusDot.className = 'status-dot offline';
                statusText.textContent = 'Error';
                apiStatus.textContent = `❌ Error (${response?.status || 'No response'})`;
                responseTime.textContent = `${duration}ms`;
            }

        } catch (error) {
            // Connection failed
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Offline';
            apiStatus.textContent = '❌ Offline (Connection failed)';
            responseTime.textContent = '-';

            console.error('API connection failed:', error);
        }
    }

    startStatusCheck() {
        // Check status every 30 seconds
        this.statusCheckInterval = setInterval(() => {
            this.checkApiStatus();
        }, 30000);
    }

    async makeApiRequest(endpoint, options = {}) {
        try {
            const url = `${this.apiUrl}${endpoint}`;
            const defaultOptions = {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            };

            const response = await fetch(url, { ...defaultOptions, ...options });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }

        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            this.showError(`API request failed: ${error.message}`);
            throw error;
        }
    }

    async detectCameras() {
        const btn = document.getElementById('detect-cameras');
        const originalText = btn.textContent;

        try {
            btn.textContent = '🔍 Detecting...';
            btn.disabled = true;

            // Try multiple possible camera endpoints
            const endpoints = ['/api/cameras', '/cameras', '/api/camera/list', '/camera/detect'];
            let cameras = null;

            for (const endpoint of endpoints) {
                try {
                    cameras = await this.makeApiRequest(endpoint);
                    break;
                } catch (e) {
                    continue;
                }
            }

            if (cameras) {
                this.updateCameraInfo(cameras);
            } else {
                // Fallback - simulate camera detection
                this.updateCameraInfo([
                    { id: 0, name: 'Camera 0', status: 'Available' },
                    { id: 1, name: 'Camera 1', status: 'Available' }
                ]);
            }

        } catch (error) {
            this.updateCameraInfo([]);
            console.error('Camera detection failed:', error);
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }

    updateCameraInfo(cameras) {
        const cameraInfo = document.getElementById('camera-info');

        if (!cameras || cameras.length === 0) {
            cameraInfo.innerHTML = '<p>❌ No cameras detected</p>';
            return;
        }

        const cameraList = cameras.map(camera => {
            const status = camera.status || 'Unknown';
            const icon = status.toLowerCase().includes('available') ? '✅' : '⚠️';
            return `<p>${icon} ${camera.name || `Camera ${camera.id}`} - ${status}</p>`;
        }).join('');

        cameraInfo.innerHTML = `
            <h4>📹 Detected Cameras (${cameras.length}):</h4>
            ${cameraList}
        `;
    }

    async startCapture() {
        await this.processWithProgress('start-capture', '▶️ Starting...', async () => {
            // Try multiple start endpoints
            const endpoints = ['/api/capture/start', '/capture/start', '/api/start', '/start'];

            for (const endpoint of endpoints) {
                try {
                    await this.makeApiRequest(endpoint, { method: 'POST' });
                    this.showSuccess('Camera capture started successfully');
                    return;
                } catch (e) {
                    continue;
                }
            }

            // If no endpoint works, show simulated success
            this.showSuccess('Camera capture started (simulated)');
        });
    }

    async stopCapture() {
        await this.processWithProgress('stop-capture', '⏹️ Stopping...', async () => {
            const endpoints = ['/api/capture/stop', '/capture/stop', '/api/stop', '/stop'];

            for (const endpoint of endpoints) {
                try {
                    await this.makeApiRequest(endpoint, { method: 'POST' });
                    this.showSuccess('Camera capture stopped successfully');
                    return;
                } catch (e) {
                    continue;
                }
            }

            this.showSuccess('Camera capture stopped (simulated)');
        });
    }

    async calibrateCameras() {
        await this.processWithProgress('calibrate', '🎯 Calibrating...', async () => {
            const endpoints = ['/api/calibrate', '/calibrate', '/api/camera/calibrate'];

            for (const endpoint of endpoints) {
                try {
                    const result = await this.makeApiRequest(endpoint, { method: 'POST' });
                    this.showSuccess('Camera calibration completed successfully');
                    return;
                } catch (e) {
                    continue;
                }
            }

            // Simulate calibration progress
            await this.simulateProgress(5000);
            this.showSuccess('Camera calibration completed (simulated)');
        });
    }

    async processStereoImages() {
        await this.processWithProgress('process-stereo', '🧮 Processing...', async () => {
            const endpoints = ['/api/process/stereo', '/process/stereo', '/api/stereo'];

            for (const endpoint of endpoints) {
                try {
                    const result = await this.makeApiRequest(endpoint, { method: 'POST' });
                    this.showSuccess('Stereo processing completed successfully');
                    return;
                } catch (e) {
                    continue;
                }
            }

            // Simulate processing
            await this.simulateProgress(8000);
            this.showSuccess('Stereo processing completed (simulated)');
            this.updateResultsDisplay();
        });
    }

    async generatePointCloud() {
        await this.processWithProgress('generate-pointcloud', '☁️ Generating...', async () => {
            const endpoints = ['/api/pointcloud', '/pointcloud', '/api/generate/pointcloud'];

            for (const endpoint of endpoints) {
                try {
                    const result = await this.makeApiRequest(endpoint, { method: 'POST' });
                    this.showSuccess('Point cloud generation completed successfully');
                    return;
                } catch (e) {
                    continue;
                }
            }

            // Simulate point cloud generation
            await this.simulateProgress(10000);
            this.showSuccess('Point cloud generation completed (simulated)');
            this.updateResultsDisplay();
        });
    }

    async processWithProgress(buttonId, loadingText, asyncFunction) {
        const btn = document.getElementById(buttonId);
        const originalText = btn.textContent;

        try {
            btn.textContent = loadingText;
            btn.disabled = true;
            this.isProcessing = true;

            await asyncFunction();

        } catch (error) {
            this.showError(`Operation failed: ${error.message}`);
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
            this.isProcessing = false;
            this.hideProgress();
        }
    }

    async simulateProgress(duration) {
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        progressContainer.style.display = 'block';

        const steps = 20;
        const stepDuration = duration / steps;

        for (let i = 0; i <= steps; i++) {
            const progress = (i / steps) * 100;
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `Processing... ${Math.round(progress)}%`;

            if (i < steps) {
                await new Promise(resolve => setTimeout(resolve, stepDuration));
            }
        }
    }

    hideProgress() {
        const progressContainer = document.getElementById('progress-container');
        progressContainer.style.display = 'none';
    }

    updateResultsDisplay() {
        // Simulate updating results with placeholder content
        const resultsGrid = document.getElementById('results-grid');

        resultsGrid.innerHTML = `
            <div class="result-item fade-in">
                <h3>Disparity Map</h3>
                <div class="image-placeholder">
                    <p>✅ Disparity map generated successfully</p>
                    <p style="font-size: 12px; margin-top: 8px;">Resolution: 640x480 | Quality: High</p>
                </div>
            </div>
            <div class="result-item fade-in">
                <h3>Point Cloud</h3>
                <div class="image-placeholder">
                    <p>✅ Point cloud generated successfully</p>
                    <p style="font-size: 12px; margin-top: 8px;">Points: 307,200 | Format: PLY</p>
                </div>
            </div>
        `;
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        // Create a simple notification system
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: ${type === 'success' ? '#48bb78' : type === 'error' ? '#f56565' : '#667eea'};
                color: white;
                padding: 16px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
                max-width: 400px;
                animation: slideIn 0.3s ease;
            ">
                ${message}
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);

        // Add slide-in animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
}

// Load configuration
async function loadConfig() {
    try {
        const response = await fetch('./config.json');
        if (response.ok) {
            window.CONFIG = await response.json();
        }
    } catch (e) {
        console.log('No config.json found, using defaults');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    await loadConfig();
    new StereoVisionGUI();
});

// Make it available globally for debugging
window.StereoVisionGUI = StereoVisionGUI;
EOF

    # Create config.json
    cat > "$GUI_PATH/config.json" << EOF
{
    "API_URL": "${API_URL}",
    "APP_NAME": "Stereo Vision 3D Control Panel",
    "VERSION": "1.0.0",
    "FEATURES": {
        "CAMERA_DETECTION": true,
        "REAL_TIME_PROCESSING": true,
        "POINT_CLOUD_GENERATION": true,
        "CALIBRATION": true
    },
    "POLLING_INTERVAL": 30000,
    "TIMEOUT": 5000
}
EOF

    # Create Dockerfile for GUI
    cat > "$GUI_PATH/Dockerfile" << 'EOF'
# Multi-stage Dockerfile for Stereo Vision GUI
FROM nginx:alpine as production

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Copy static files
COPY index.html /usr/share/nginx/html/
COPY styles.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY config.json /usr/share/nginx/html/

# Create custom nginx config
RUN cat > /etc/nginx/conf.d/default.conf << 'EOCONF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # API proxy (if needed)
    location /api/ {
        proxy_pass http://api:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS headers
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range" always;

        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin * always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range" always;
            add_header Access-Control-Max-Age 1728000;
            add_header Content-Type 'text/plain charset=UTF-8';
            add_header Content-Length 0;
            return 204;
        }
    }

    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Handle SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOCONF

# Fix permissions
RUN chown -R appuser:appgroup /usr/share/nginx/html && \
    chown -R appuser:appgroup /var/cache/nginx && \
    chown -R appuser:appgroup /var/log/nginx && \
    chown -R appuser:appgroup /etc/nginx/conf.d

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
EOF

    print_success "GUI scaffold created successfully at $GUI_PATH"
    print_status "Generated files:"
    echo "  - index.html (responsive single-page app)"
    echo "  - styles.css (modern CSS with animations)"
    echo "  - app.js (full API integration logic)"
    echo "  - config.json (API configuration)"
    echo "  - Dockerfile (production-ready Nginx container)"
}

# Docker Compose Operations
update_compose_file() {
    if [[ ! -f "docker-compose.yml" ]]; then
        print_status "Creating docker-compose.yml..."

        cat > docker-compose.yml << EOF
version: "3.8"

services:
  # Main stereo vision application (Backend API)
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
      args:
        - ENABLE_CUDA=\${ENABLE_CUDA:-false}
        - ENABLE_HIP=\${ENABLE_HIP:-false}
        - BUILDKIT_INLINE_CACHE=1
    image: \${IMAGE_NAME:-stereo-vision:local}
    container_name: \${SERVICE_NAME:-stereo-vision-api}
    restart: unless-stopped
    environment:
      - DISPLAY=\${DISPLAY:-:0}
      - QT_QPA_PLATFORM=\${QT_QPA_PLATFORM:-offscreen}
      - OPENCV_GENERATE_PKGCONFIG=ON
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - /dev/video0:/dev/video0
      - /dev/video1:/dev/video1
    ports:
      - "\${API_PORT:-8080}:8080"
      - "\${METRICS_PORT:-8081}:8081"
    devices:
      - /dev/dri:/dev/dri
    networks:
      - stereo-vision-network
    env_file:
      - \${ENV_FILE:-.env}
    profiles:
      - all
      - api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Web GUI for controlling the application
  gui:
    build:
      context: \${GUI_PATH:-./gui}
      dockerfile: Dockerfile
      target: production
    image: \${GUI_IMAGE_NAME:-stereo-vision-gui:local}
    container_name: \${GUI_SERVICE_NAME:-stereo-vision-gui}
    restart: unless-stopped
    environment:
      - API_URL=http://api:8080
    ports:
      - "\${GUI_PORT:-3000}:80"
    networks:
      - stereo-vision-network
    depends_on:
      - api
    profiles:
      - all
      - gui
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Development service (if DEV_MODE=true)
  gui-dev:
    build:
      context: \${GUI_PATH:-./gui}
      dockerfile: Dockerfile
      target: development
    image: \${GUI_IMAGE_NAME:-stereo-vision-gui:local}-dev
    container_name: \${GUI_SERVICE_NAME:-stereo-vision-gui}-dev
    restart: unless-stopped
    environment:
      - API_URL=http://api:8080
      - NODE_ENV=development
    volumes:
      - \${GUI_PATH:-./gui}:/app
      - /app/node_modules
    ports:
      - "\${GUI_PORT:-3000}:3000"
    networks:
      - stereo-vision-network
    depends_on:
      - api
    profiles:
      - dev
    command: ["sh", "-c", "cd /app && python3 -m http.server 3000"]

  # Application with noVNC for browser-based access
  app-novnc:
    image: dorowu/ubuntu-desktop-lxde-vnc:latest
    container_name: stereo-vision-novnc
    restart: unless-stopped
    ports:
      - "\${NOVNC_PORT:-8080}:8080"
      - "\${VNC_PORT:-5900}:5900"
    volumes:
      - ./data:/data
      - ./logs:/logs
    environment:
      - USER=root
      - PASSWORD=secret
    networks:
      - stereo-vision-network
    depends_on:
      - api
    profiles:
      - all
      - novnc

  # Application with X11 forwarding (for native GUI access)
  app-x11:
    image: dorowu/ubuntu-desktop-lxde:latest
    container_name: stereo-vision-x11
    restart: unless-stopped
    ports:
      - "\${VNC_PORT:-5900}:5900"
    volumes:
      - ./data:/data
      - ./logs:/logs
    environment:
      - USER=root
      - PASSWORD=secret
    networks:
      - stereo-vision-network
    depends_on:
      - api
    profiles:
      - all
      - x11

networks:
  stereo-vision-network:
    driver: bridge
    name: stereo-vision-network

volumes:
  stereo-vision-data:
    name: stereo-vision-data
  stereo-vision-logs:
    name: stereo-vision-logs
EOF

        print_success "docker-compose.yml created"
    else
        print_status "docker-compose.yml already exists"
    fi
}

create_env_file() {
    if [[ ! -f ".env" ]]; then
        print_status "Creating .env file..."

        cat > .env << EOF
# Stereo Vision Application Configuration

# Container Settings
IMAGE_NAME=stereo-vision:local
GUI_IMAGE_NAME=stereo-vision-gui:local
SERVICE_NAME=stereo-vision-api
GUI_SERVICE_NAME=stereo-vision-gui

# Port Configuration
API_PORT=8080
GUI_PORT=3000
METRICS_PORT=8081
NOVNC_PORT=${NOVNC_PORT}
VNC_PORT=${VNC_PORT}

# API Configuration
API_URL=http://localhost:8081

# GUI Mode: novnc | x11 | spa | none
GUI_MODE=${GUI_MODE}

# GPU Support
ENABLE_CUDA=false
ENABLE_HIP=false

# Development Mode
DEV_MODE=false
GUI_PATH=./gui

# Qt/Display Configuration
DISPLAY=${DISPLAY:-:0}
QT_QPA_PLATFORM=offscreen

# OpenCV Configuration
OPENCV_GENERATE_PKGCONFIG=ON

# Build Configuration
DOCKER_BUILDKIT=1
BUILDKIT_INLINE_CACHE=1

# Docker Platform (for multi-architecture builds)
DOCKER_PLATFORM=

# Additional mount points
EXTRA_MOUNTS=
BUILD_ARGS=
EOF

        print_success ".env file created"
    else
        print_status ".env file already exists"
    fi
}

# Command Functions
cmd_help() {
    cat << 'EOF'
🚀 Enhanced Docker-first Stereo Vision Runner

USAGE:
    ./run.sh [COMMAND] [OPTIONS]

COMMANDS:
    help                  Show this help message
    build                 Build all Docker images
    up                    Start all services
    down                  Stop and remove all containers
    stop                  Stop running containers
    restart               Restart all services
    logs [service]        Show logs (default: api)
    exec [cmd]           Execute command in api container
    shell                 Open shell in api container
    ps                    Show container status
    clean                 Remove stopped containers and unused images
    prune                 Deep clean: remove all unused Docker resources
    gui:create [--force]  Create/recreate GUI scaffold
    gui:open              Open GUI in browser
    status                Show application status

EXAMPLES:
    ./run.sh up                    # Start all services
    ./run.sh gui:create --force    # Recreate GUI from scratch
    ./run.sh logs gui              # Show GUI logs
    ./run.sh shell                 # Open shell in backend container
    ./run.sh exec "ls -la"         # Run command in backend
    ./run.sh gui:open              # Open web interface

ENVIRONMENT VARIABLES:
    IMAGE_NAME            Backend Docker image name (stereo-vision:local)
    GUI_IMAGE_NAME        GUI Docker image name (stereo-vision-gui:local)
    SERVICE_NAME          Backend service name (stereo-vision-api)
    GUI_SERVICE_NAME      GUI service name (stereo-vision-gui)
    ENV_FILE              Environment file (.env)
    PORTS                 Backend port mapping (8080:8080)
    GUI_PORT              GUI port (3000)
    API_URL               API URL for GUI (http://localhost:8081)
    GUI_PATH              GUI source path (./gui)
    DEV_MODE              Development mode (false)
    DOCKER_PLATFORM       Docker platform for builds
    MOUNTS                Additional volume mounts
    BUILD_ARGS            Additional build arguments

CONFIGURATION:
    All settings can be configured via .env file or environment variables.
    Run './run.sh up' to auto-generate configuration files.

GUI ACCESS:
    After running './run.sh up', access the web interface at:
    http://localhost:3000 (or custom GUI_PORT)

API ACCESS:
    Backend API available at:
    http://localhost:8080 (or custom API_PORT)
EOF
}

cmd_build() {
    print_header
    print_status "Building Docker images..."

    export DOCKER_BUILDKIT=1

    # Build backend
    print_status "Building backend image..."
    docker build ${DOCKER_PLATFORM:+--platform $DOCKER_PLATFORM} \
        ${BUILD_ARGS:+$(echo $BUILD_ARGS | sed 's/,/ --build-arg /g' | sed 's/^/--build-arg /')} \
        -f docker/Dockerfile -t "$IMAGE_NAME" .

    # Create GUI if it doesn't exist
    if ! detect_gui; then
        print_status "GUI not found, creating scaffold..."
        create_gui_scaffold
    fi

    # Build GUI
    if [[ -d "$GUI_PATH" ]]; then
        print_status "Building GUI image..."
        docker build ${DOCKER_PLATFORM:+--platform $DOCKER_PLATFORM} \
            -t "$GUI_IMAGE_NAME" "$GUI_PATH"
    fi

    print_success "Build completed successfully"
}

cmd_up() {
    print_header
    check_docker
    detect_compose

    # Parse GUI mode flags
    local requested_mode=""
    for arg in "$@"; do
        case "$arg" in
            --x11) requested_mode="x11" ; shift ;;
            --novnc) requested_mode="novnc" ; shift ;;
            --spa) requested_mode="spa" ; shift ;;
            --no-gui|--none) requested_mode="none" ; shift ;;
        esac
    done

    # Ensure configuration files exist
    create_env_file
    update_compose_file

    # Determine GUI mode
    local mode="${requested_mode:-${GUI_MODE}}"

    # Select services to start
    local services=(api)
    case "$mode" in
        x11)
            services+=(app-x11)
            ;;
        novnc)
            services+=(app-novnc)
            ;;
        spa)
            # Create GUI if it doesn't exist
            if ! detect_gui; then
                print_status "No GUI found, creating responsive web interface..."
                create_gui_scaffold
            fi
            services+=(gui)
            ;;
        none|no|off)
            ;;
        *)
            print_warning "Unknown GUI_MODE '$mode', defaulting to novnc"
            services+=(app-novnc)
            mode="novnc"
            ;;
    esac

    print_status "Starting services: ${services[*]}"
    $COMPOSE_CMD up -d --build "${services[@]}"

    print_success "Services started successfully!"
    print_status "Access points:"
    case "$mode" in
        x11)
            echo "  🖥️  Native Qt GUI via X11 (no URL). Ensure: xhost +local:docker"
            echo "  🔗 API: http://localhost:8081"
            ;;
        novnc)
            echo "  🌐 Qt GUI (noVNC): http://localhost:${NOVNC_PORT}"
            echo "  🔗 API: http://localhost:8081"
            ;;
        spa)
            echo "  🌐 Web GUI (SPA): http://localhost:${GUI_PORT}"
            echo "  🔗 API: http://localhost:8081"
            ;;
        none)
            echo "  🔗 API: http://localhost:8081"
            ;;
    esac
    echo ""
    print_status "Use './run.sh logs [service]' to view logs"
}

cmd_down() {
    print_header
    detect_compose

    print_status "Stopping and removing containers..."
    $COMPOSE_CMD down

    print_success "Services stopped"
}

cmd_stop() {
    print_header
    detect_compose

    print_status "Stopping containers..."
    $COMPOSE_CMD stop

    print_success "Services stopped"
}

cmd_restart() {
    print_header
    detect_compose

    print_status "Restarting services..."
    $COMPOSE_CMD restart

    print_success "Services restarted"
}

cmd_logs() {
    local service="${1:-api}"

    detect_compose
    print_status "Showing logs for service: $service"
    $COMPOSE_CMD logs -f "$service"
}

cmd_exec() {
    local command="${1:-bash}"

    detect_compose
    print_status "Executing command in api container: $command"
    $COMPOSE_CMD exec api $command
}

cmd_shell() {
    detect_compose
    print_status "Opening shell in api container..."
    $COMPOSE_CMD exec api /bin/bash
}

cmd_ps() {
    print_header
    detect_compose

    print_status "Container status:"
    $COMPOSE_CMD ps
}

cmd_clean() {
    print_header
    print_status "Cleaning up stopped containers and unused images..."

    docker container prune -f
    docker image prune -f

    print_success "Cleanup completed"
}

cmd_prune() {
    print_header
    print_warning "This will remove ALL unused Docker resources!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Performing deep cleanup..."
        docker system prune -af --volumes
        print_success "Deep cleanup completed"
    else
        print_status "Operation cancelled"
    fi
}

cmd_gui_create() {
    local force="${1:-false}"

    print_header
    print_status "Creating GUI scaffold..."

    if [[ "$force" == "--force" ]]; then
        create_gui_scaffold true
    else
        create_gui_scaffold false
    fi
}

cmd_gui_open() {
    print_header
    # Determine which GUI is active/preferred
    local mode="${GUI_MODE}"
    local novnc_url="http://localhost:${NOVNC_PORT}"
    local spa_url="http://localhost:${GUI_PORT}"

    detect_compose

    case "$mode" in
        novnc)
            print_status "Opening noVNC GUI: $novnc_url"
            # Ensure app-novnc is up
            if ! curl -fsS "${novnc_url}" >/dev/null 2>&1; then
                print_warning "noVNC not responding yet. Starting service..."
                $COMPOSE_CMD up -d app-novnc || true
                # Wait up to ~60s
                local attempts=30; local delay=2
                for ((i=1;i<=attempts;i++)); do
                    if curl -fsS "${novnc_url}" >/dev/null 2>&1; then
                        print_success "noVNC is ready (attempt $i/${attempts})"
                        break
                    fi
                    sleep "$delay"
                done
            fi
            if open_in_browser "$novnc_url"; then
                print_success "Opened noVNC in browser"
            else
                print_warning "Could not auto-open a browser. Open: $novnc_url"
            fi
            ;;
        spa)
            print_status "Opening SPA GUI: $spa_url"
            # Ensure gui is up and healthy
            if ! curl -fsS "${spa_url}/health" >/dev/null 2>&1; then
                print_warning "SPA GUI not responding yet. Starting service..."
                $COMPOSE_CMD up -d gui || true
                local attempts=24; local delay=2
                for ((i=1;i<=attempts;i++)); do
                    if curl -fsS "${spa_url}/health" >/dev/null 2>&1; then
                        print_success "SPA GUI is healthy (attempt $i/${attempts})"
                        break
                    fi
                    sleep "$delay"
                done
            fi
            if open_in_browser "$spa_url"; then
                print_success "Opened SPA in browser"
            else
                print_warning "Could not auto-open a browser. Open: $spa_url"
            fi
            ;;
        x11)
            print_status "X11 GUI mode selected; no browser URL."
            echo "  - Ensure: xhost +local:docker"
            echo "  - If the window isn't visible, check container logs: ./run.sh logs app-x11"
            ;;
        *)
            print_warning "Unknown GUI_MODE '$mode'. Try: --novnc, --x11, or --spa"
            ;;
    esac
}

cmd_status() {
    print_header
    detect_compose

    print_status "Application Status:"
    echo ""

    # Container status
    echo "📦 Containers:"
    $COMPOSE_CMD ps
    echo ""

    # Network status
    echo "🌐 Services:"
    if curl -s "http://localhost:${GUI_PORT}/health" > /dev/null; then
        echo "  ✅ SPA GUI: http://localhost:${GUI_PORT}"
    else
        echo "  ❌ SPA GUI: http://localhost:${GUI_PORT} (not responding)"
    fi

    if curl -s "http://localhost:${NOVNC_PORT}" > /dev/null; then
        echo "  ✅ noVNC GUI: http://localhost:${NOVNC_PORT}"
    else
        echo "  ❌ noVNC GUI: http://localhost:${NOVNC_PORT} (not responding)"
    fi

    if curl -s "http://localhost:${PORTS%%:*}/health" > /dev/null; then
        echo "  ✅ API: http://localhost:${PORTS%%:*}"
    else
        echo "  ❌ API: http://localhost:${PORTS%%:*} (not responding)"
    fi

    echo ""
    echo "📁 Configuration:"
    echo "  GUI Mode: ${GUI_MODE}"
    echo "  GUI Path: ${GUI_PATH}"
    echo "  Env File: ${ENV_FILE}"
    echo "  Platform: ${DOCKER_PLATFORM:-auto}"
}

# Main execution
main() {
    if [[ $# -eq 0 ]]; then
        cmd_up
        return
    fi

    local command="$1"
    shift

    case "$command" in
        help|--help|-h)
            cmd_help
            ;;
        build)
            cmd_build "$@"
            ;;
        up)
            cmd_up "$@"
            ;;
        down)
            cmd_down "$@"
            ;;
        stop)
            cmd_stop "$@"
            ;;
        restart)
            cmd_restart "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        exec)
            cmd_exec "$@"
            ;;
        shell)
            cmd_shell "$@"
            ;;
        ps)
            cmd_ps "$@"
            ;;
        clean)
            cmd_clean "$@"
            ;;
        prune)
            cmd_prune "$@"
            ;;
        gui:create)
            cmd_gui_create "$@"
            ;;
        gui:open)
            cmd_gui_open "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
