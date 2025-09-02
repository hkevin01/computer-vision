# Docker Usage Guide for Stereo Vision Application

## Quick Start with Docker

Your stereo vision application now supports Docker for consistent cross-platform deployment and development.

### 1. Setup Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env to configure your setup (optional)
# - Set ENABLE_CUDA=true for NVIDIA GPU support
# - Set ENABLE_HIP=true for AMD GPU support
# - Customize ports and image names as needed
```

### 2. Build and Run

```bash
# Build the Docker images
docker compose build

# Start the application
docker compose up -d

# View logs
docker compose logs -f

# Stop the application
docker compose down
```

## Benefits

- One-command startup with Docker Compose and ./run.sh helper.
- Browser-based GUI (noVNC) option that works cross-platform; native X11 GUI on Linux.
- Consistent dev/prod environments using multi-stage images and profiles.
- GPU-ready builds via ENABLE_CUDA / ENABLE_HIP flags.
- Persistent data and logs: host ./data and ./logs are bind-mounted and survive rebuilds/restarts.

### 3. Development Mode

```bash
# Start development environment with live code reload
docker compose --profile dev up -d

# Open an interactive shell in the container
docker compose exec stereo-vision-dev bash

# Run tests inside the container
docker compose exec stereo-vision-app bash -c "cd /app/build && ctest"
```

### 4. Enhanced run.sh Commands

The `run.sh` script maintains backward compatibility while adding Docker support:

```bash
# Traditional native build (still works)
./run.sh --build-only
./run.sh --status

# New Docker commands (when implemented)
./run.sh build       # Build Docker images
./run.sh up          # Start Docker application
./run.sh dev         # Development mode
./run.sh shell       # Interactive shell
./run.sh status      # Show Docker status
```

### 5. GPU Acceleration

For NVIDIA GPU support:

```bash
# Set environment variable and build
ENABLE_CUDA=true docker compose build

# Uncomment GPU sections in docker-compose.yml
# Then start normally
docker compose up -d
```

For AMD GPU support:

```bash
# Set environment variable and build
ENABLE_HIP=true docker compose build
docker compose up -d
```

### 6. GUI Applications

For GUI applications requiring X11:

```bash
# Ensure X11 is available
export DISPLAY=:0

# Enable GUI mode in docker-compose.yml by setting:
# QT_QPA_PLATFORM=xcb

# Start with GUI support
docker compose up -d
```

For headless operation:

```bash
# Use offscreen rendering
export QT_QPA_PLATFORM=offscreen
docker compose up -d
```

### 7. Services Available

- **stereo-vision-app**: Production application with optimized build
- **stereo-vision-dev**: Development environment with debugging tools
- **stereo-vision-simple**: Lightweight version with minimal dependencies

### 8. Ports and Access

Default port mappings:

- **8080**: Application API or browser GUI (noVNC), depending on profile
- **8081**: Metrics/health when API is enabled
- **5900**: VNC (optional, when using noVNC profile)

### 9. Data and Volumes

The application mounts these directories by default:

- `./data:/app/data:rw` - Application datasets/configuration
- `./logs:/app/logs:rw` - Runtime logs from the application
- `/tmp/.X11-unix:/tmp/.X11-unix:rw` - X11 socket (only for native X11 GUI)

Note: The `./data` and `./logs` directories are on the host and persist across container rebuilds and restarts. You can safely wipe logs with `rm -rf logs/*` without affecting data.

### 10. Troubleshooting

**Docker not found:**

```bash
# Install Docker Desktop or Docker Engine
# Add user to docker group: sudo usermod -aG docker $USER
```

**Permission denied:**

```bash
# Run with sudo or add user to docker group
sudo docker compose up -d
```

**GUI not working:**

```bash
# Ensure X11 forwarding is enabled
xhost +local:docker
export DISPLAY=:0
```

**Build failures:**

```bash
# Clean and rebuild
docker compose down
docker system prune -f
docker compose build --no-cache
```

This Docker setup provides a consistent, isolated environment for your stereo vision application while maintaining full compatibility with existing native builds.
