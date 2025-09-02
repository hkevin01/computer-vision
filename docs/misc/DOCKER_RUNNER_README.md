# 🚀 Enhanced Docker-first Stereo Vision Runner

## Overview

This enhanced `run.sh` script provides a comprehensive Docker-first approach to building and running your C++ Stereo Vision application with an auto-generated web GUI for control and monitoring.

## Features

- **🐳 Docker-first Architecture**: Full containerization with multi-stage builds
- **🌐 Auto-Generated Web GUI**: Responsive web interface if no GUI exists
- **🔧 Multiple Operation Modes**: API server, native GUI, or simple mode
- **📊 Real-time Monitoring**: Live status updates and performance metrics
- **🎯 Cross-platform Support**: Works on Linux and macOS
- **⚡ GPU Acceleration**: Optional CUDA/HIP support
- **🛠️ Development Tools**: Built-in debugging and development modes

## Quick Start

### 1. Initial Setup

```bash
# Clone or navigate to your project
cd /path/to/stereo-vision

# Make run.sh executable
chmod +x run.sh

# Start the application (auto-generates config files)
./run.sh up
```

### 2. Access Points

After running `./run.sh up`:

- **Web GUI**: http://localhost:3000
- **API**: http://localhost:8080
- **Metrics**: http://localhost:8081

## Command Reference

### Basic Commands

```bash
./run.sh help                  # Show comprehensive help
./run.sh build                 # Build all Docker images
./run.sh up                    # Start all services
./run.sh down                  # Stop and remove containers
./run.sh restart               # Restart all services
./run.sh status                # Show application status
```

### GUI Management

```bash
./run.sh gui:create            # Create GUI scaffold
./run.sh gui:create --force    # Recreate GUI from scratch
./run.sh gui:open              # Open web interface in browser
```

### Development & Debugging

```bash
./run.sh logs                  # Show API logs
./run.sh logs gui              # Show GUI logs
./run.sh shell                 # Open shell in API container
./run.sh exec "command"        # Execute command in container
```

### Maintenance

```bash
./run.sh clean                 # Remove stopped containers
./run.sh prune                 # Deep clean all unused resources
./run.sh ps                    # Show container status
```

## Configuration

### Environment Variables

All configuration is managed via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_NAME` | stereo-vision:local | Backend Docker image name |
| `GUI_IMAGE_NAME` | stereo-vision-gui:local | GUI Docker image name |
| `SERVICE_NAME` | stereo-vision-api | Backend service name |
| `GUI_SERVICE_NAME` | stereo-vision-gui | GUI service name |
| `API_PORT` | 8080 | Backend API port |
| `GUI_PORT` | 3000 | Web GUI port |
| `API_URL` | http://localhost:8080 | API URL for GUI |
| `GUI_PATH` | ./gui | GUI source directory |
| `DEV_MODE` | false | Enable development mode |
| `ENABLE_CUDA` | false | Enable NVIDIA GPU support |
| `ENABLE_HIP` | false | Enable AMD GPU support |

### Configuration Files

The script auto-generates these files if they don't exist:

- `.env` - Environment configuration
- `docker-compose.yml` - Service orchestration
- `./gui/` - Web interface files (if missing)

## GUI Features

The auto-generated web GUI provides:

### 🎛️ Control Panel
- **API Status Monitoring**: Real-time connection status
- **Camera Management**: Detect and configure cameras
- **Processing Controls**: Calibration, stereo processing, point cloud generation

### 📊 Real-time Feedback
- **Live Status Updates**: Connection quality and response times
- **Progress Tracking**: Visual progress bars for operations
- **Results Display**: Preview generated depth maps and point clouds

### 📱 Responsive Design
- **Mobile-friendly**: Works on phones, tablets, and desktops
- **Modern UI**: Glass-morphism design with smooth animations
- **Cross-browser**: Compatible with all modern browsers

## Architecture

### Backend (C++ Application)

- **Native GUI Mode**: Runs your Qt5 stereo vision application
- **API Mode**: Exposes REST endpoints for web control
- **Simple Mode**: Runs simplified version with fewer dependencies

### Frontend (Web GUI)

- **Static Files**: Single-page application with no build dependencies
- **Real-time Communication**: AJAX-based API calls with error handling
- **Progressive Enhancement**: Works with or without active backend

## Development Workflow

### 1. Standard Development

```bash
# Start in development mode
DEV_MODE=true ./run.sh up

# Watch logs
./run.sh logs

# Execute commands
./run.sh exec "ls -la"
```

### 2. GUI Development

```bash
# Recreate GUI with custom changes
./run.sh gui:create --force

# Edit files in ./gui/
# Changes are reflected immediately in development mode
```

### 3. API Development

```bash
# Test API endpoints
curl http://localhost:8080/health
curl http://localhost:8080/api/cameras

# View detailed logs
./run.sh logs api
```

## Troubleshooting

### Common Issues

**1. Docker daemon not running**
```bash
# Start Docker service
sudo systemctl start docker
```

**2. Permission issues**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

**3. Port conflicts**
```bash
# Check what's using ports
ss -tulpn | grep :3000
ss -tulpn | grep :8080

# Change ports in .env file
echo "GUI_PORT=3001" >> .env
echo "API_PORT=8081" >> .env
```

**4. GPU support issues**
```bash
# Check NVIDIA setup
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check AMD setup
rocm-smi
```

### Debug Mode

```bash
# Enable verbose output
DOCKER_BUILDKIT=1 ./run.sh build

# Check container health
./run.sh ps
docker logs stereo-vision-api
docker logs stereo-vision-gui
```

## Performance Optimization

### Resource Allocation

```bash
# Limit container resources
echo "DOCKER_MEMORY=2g" >> .env
echo "DOCKER_CPUS=2" >> .env
```

### Build Optimization

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Multi-stage build caching
./run.sh build
```

## Advanced Configuration

### Custom Build Arguments

```bash
# Pass build arguments
BUILD_ARGS="OPENCV_VERSION=4.8.0,CUDA_VERSION=11.8" ./run.sh build
```

### Custom Mounts

```bash
# Mount additional directories
MOUNTS="/host/data:/container/data,/host/models:/container/models" ./run.sh up
```

### Network Configuration

```bash
# Custom network settings
docker network create stereo-vision-network --driver bridge
```

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/docker.yml
name: Docker Build and Test
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and test
        run: |
          chmod +x run.sh
          ./run.sh build
          ./run.sh up -d
          sleep 30
          curl -f http://localhost:3000/health
          curl -f http://localhost:8080/health
          ./run.sh down
```

### Production Deployment

```bash
# Production environment
cat > .env.prod << EOF
IMAGE_NAME=stereo-vision:production
GUI_IMAGE_NAME=stereo-vision-gui:production
API_PORT=80
GUI_PORT=443
ENABLE_CUDA=true
DEV_MODE=false
EOF

# Deploy
ENV_FILE=.env.prod ./run.sh up
```

## Contributing

### Adding New Features

1. **Backend**: Modify your C++ application and update Dockerfile
2. **API**: Add endpoints to the Python API server in entrypoint.sh
3. **GUI**: Update GUI files in `./gui/` directory
4. **Documentation**: Update this README with new features

### Testing

```bash
# Run full test suite
./run.sh build
./run.sh up -d
./test_integration.sh
./run.sh down
```

## Support

### Getting Help

- **Documentation**: Check this README and `./run.sh help`
- **Logs**: Use `./run.sh logs` for debugging
- **Status**: Use `./run.sh status` for health checks

### Reporting Issues

When reporting issues, include:

1. Output of `./run.sh status`
2. Relevant logs from `./run.sh logs`
3. Your `.env` configuration (redacted)
4. Docker version: `docker --version`
5. System information: `uname -a`

---

**🎉 You now have a fully containerized, web-enabled stereo vision application with professional Docker orchestration!**
