# Multi-stage Dockerfile for Stereo Vision 3D Point Cloud Application
FROM ubuntu:22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies (Qt6)
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    # C++ dependencies
    libopencv-dev \
    libpcl-dev \
    libeigen3-dev \
    # Qt6 dependencies
    qt6-base-dev \
    qt6-base-dev-tools \
    qt6-tools-dev \
    # X11 and GUI dependencies for headless/desktop operation
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxss-dev \
    libx11-xcb1 \
    libxkbcommon-x11-0 \
    # Audio (for potential multimedia support)
    libasound2-dev \
    # Additional OpenCV dependencies
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    # Threading and parallel processing
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Development stage with additional tools
FROM base AS development

# Install development tools and web server dependencies
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    vim \
    nano \
    htop \
    tree \
    curl \
    wget \
    # Web server dependencies for API mode
    python3 \
    python3-pip \
    python3-dev \
    nginx \
    # Additional networking tools
    netcat-openbsd \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for potential web API
RUN pip3 install --no-cache-dir \
    flask \
    fastapi \
    uvicorn \
    requests \
    psutil

# GPU support stage (optional, can be built with --target gpu)
FROM base AS gpu

# Install NVIDIA CUDA support (conditional)
ARG ENABLE_CUDA=false
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
    apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Install AMD ROCm support (conditional)
ARG ENABLE_HIP=false
RUN if [ "$ENABLE_HIP" = "true" ]; then \
    apt-get update && apt-get install -y \
    rocm-dev \
    hip-dev \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Production stage
FROM base AS production

# Ensure python3 available for API mode and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appgroup && useradd -r -g appgroup -m -d /home/appuser -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY CMakeLists.txt ./
COPY cmake/ ./cmake/

# Copy source code
COPY include/ ./include/
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY test_programs/ ./test_programs/

# Set up build environment
ENV CMAKE_BUILD_TYPE=Release
ENV BUILD_DIR=build
ENV OPENCV_GENERATE_PKGCONFIG=ON

# Build the application
RUN mkdir -p build && \
    cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_GUI=ON \
    -DBUILD_TESTS=OFF \
    -DUSE_CUDA=OFF \
    -DUSE_HIP=OFF \
    -DWITH_OPENMP=ON \
    -GNinja && \
    ninja && \
    # Clean up build artifacts to reduce image size
    rm -rf CMakeFiles/ cmake_install.cmake CMakeCache.txt

# Copy and set up entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Change ownership to app user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose potential ports (adjustable)
EXPOSE 8080 8081

# Health check that works for both modes
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD bash -lc "if [ \"$APP_MODE\" = \"api\" ]; then python3 -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8080/health\")' >/dev/null 2>&1 || exit 1; else test -f /app/build/stereo_vision_app || exit 1; fi"

# Default command
CMD ["/app/entrypoint.sh"]

# Multi-target support
FROM production AS simple
CMD ["./build/stereo_vision_app_simple"]

FROM development AS dev
USER root
WORKDIR /app
CMD ["/bin/bash"]
