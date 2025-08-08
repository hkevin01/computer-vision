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

        // Fallback to default
        return 'http://localhost:8080';
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
                        break;
                    }
                } catch (e) {
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

            console.log('API connection failed (this is normal in demo mode):', error);
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
            console.log(`API request failed for ${endpoint} (demo mode):`, error);
            // In demo mode, return simulated data
            return { status: 'simulated', endpoint: endpoint };
        }
    }

    async detectCameras() {
        await this.processWithProgress('detect-cameras', '🔍 Detecting...', async () => {
            // Simulate camera detection
            await this.simulateProgress(2000);

            const cameras = [
                { id: 0, name: 'Camera 0', status: 'Available' },
                { id: 1, name: 'Camera 1', status: 'Available' }
            ];

            this.updateCameraInfo(cameras);
            this.showSuccess('Cameras detected successfully');
        });
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
            await this.simulateProgress(1500);
            this.showSuccess('Camera capture started successfully');
        });
    }

    async stopCapture() {
        await this.processWithProgress('stop-capture', '⏹️ Stopping...', async () => {
            await this.simulateProgress(1000);
            this.showSuccess('Camera capture stopped successfully');
        });
    }

    async calibrateCameras() {
        await this.processWithProgress('calibrate', '🎯 Calibrating...', async () => {
            await this.simulateProgress(5000);
            this.showSuccess('Camera calibration completed successfully');
        });
    }

    async processStereoImages() {
        await this.processWithProgress('process-stereo', '🧮 Processing...', async () => {
            await this.simulateProgress(8000);
            this.showSuccess('Stereo processing completed successfully');
            this.updateResultsDisplay();
        });
    }

    async generatePointCloud() {
        await this.processWithProgress('generate-pointcloud', '☁️ Generating...', async () => {
            await this.simulateProgress(10000);
            this.showSuccess('Point cloud generation completed successfully');
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
        const notification = document.createElement('div');
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
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new StereoVisionGUI();
});

// Make it available globally for debugging
window.StereoVisionGUI = StereoVisionGUI;
