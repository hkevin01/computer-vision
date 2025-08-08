# Docker Setup for Stereo Vision Application

## Quick Setup

The project has been enhanced with Docker support for easier deployment and development. Here are the key files that have been created:

### Files Created

1. **Dockerfile** - Multi-stage Docker build configuration
2. **docker-compose.yml** - Docker Compose orchestration
3. **.env.example** - Environment configuration template

### Enhanced run.sh Commands

The run.sh script now supports Docker-first operation with these new commands:

```bash
# Docker Commands
./run.sh build           # Build Docker images
./run.sh up              # Start the application
./run.sh down            # Stop and remove containers
./run.sh logs            # View application logs
./run.sh shell           # Open shell in container
./run.sh exec "command"  # Execute command in container

# Development Commands
./run.sh dev             # Start development environment
./run.sh test            # Run tests in container
./run.sh debug           # Start debugging session

# Status and Utilities
./run.sh status          # Show project and container status
./run.sh gui             # Launch GUI version (requires X11)
```

### Quick Start Guide

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Build and run:**
   ```bash
   ./run.sh build
   ./run.sh up
   ```

3. **For development with live reload:**
   ```bash
   MOUNTS=".:/app" ./run.sh dev
   ```

4. **For GPU support (NVIDIA):**
   ```bash
   ENABLE_CUDA=true ./run.sh build
   ```

### Environment Variables

Key configuration options in `.env`:

```bash
# Image and service names
IMAGE_NAME=stereo-vision:local
SERVICE_NAME=stereo-vision-app

# Port mappings
PORTS=8080:8080,8081:8081

# GPU support
ENABLE_CUDA=false
ENABLE_HIP=false

# Development
MOUNTS=.:/app
BUILD_ARGS=CMAKE_BUILD_TYPE=Debug
```

### Docker Compose Services

- **stereo-vision-app**: Main application (production)
- **stereo-vision-dev**: Development environment with tools
- **stereo-vision-simple**: Lightweight version

### Legacy Native Build Support

The original native build commands are still available:

```bash
./run.sh native-build    # Build using native CMake
./run.sh native-run      # Run native build
./run.sh native-clean    # Clean native build
./run.sh native-test     # Run native tests
```

### Benefits of Docker Approach

1. **Consistent Environment**: Same runtime across all systems
2. **No Dependency Conflicts**: Isolated from host system libraries
3. **Easy Distribution**: Share complete application environment
4. **Development Efficiency**: Quick setup for new developers
5. **GPU Support**: Optional CUDA/HIP acceleration
6. **Multiple Variants**: Production, development, and simple builds

### GUI Support

For GUI applications with X11:

```bash
# Set display and enable GUI
export DISPLAY=:0
./run.sh gui
```

For headless operation:
```bash
export QT_QPA_PLATFORM=offscreen
./run.sh up
```

### Troubleshooting

1. **Docker not found**: Install Docker Desktop or Docker Engine
2. **Permission denied**: Add user to docker group or use sudo
3. **GUI not working**: Ensure X11 forwarding is enabled
4. **Build failures**: Check .env file and build arguments

This Docker setup maintains full compatibility with the existing project while adding modern containerization capabilities.
