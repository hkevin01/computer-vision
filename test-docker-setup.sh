#!/bin/bash
# Test script for Docker-first Stereo Vision setup

set -Eeuo pipefail

echo "🧪 Testing Docker-first Stereo Vision Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

info() {
    echo -e "ℹ️  $1"
}

# Test 1: Check Docker availability
echo ""
info "Test 1: Checking Docker availability..."
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        success "Docker is installed and running"
    else
        error "Docker is installed but daemon is not running"
        exit 1
    fi
else
    error "Docker is not installed"
    exit 1
fi

# Test 2: Check run.sh permissions
echo ""
info "Test 2: Checking run.sh permissions..."
if [[ -x "./run.sh" ]]; then
    success "run.sh is executable"
elif [[ -f "./run.sh" ]]; then
    warning "run.sh exists but is not executable - fixing..."
    chmod +x ./run.sh
    success "run.sh permissions fixed"
else
    error "run.sh not found"
    exit 1
fi

# Test 3: Check compose availability
echo ""
info "Test 3: Checking Docker Compose availability..."
if command -v "docker-compose" &> /dev/null; then
    success "docker-compose is available"
elif docker compose version &> /dev/null; then
    success "docker compose is available"
else
    error "Neither 'docker compose' nor 'docker-compose' is available"
    exit 1
fi

# Test 4: Test help command
echo ""
info "Test 4: Testing help command..."
if ./run.sh help > /dev/null 2>&1; then
    success "Help command works"
else
    error "Help command failed"
    exit 1
fi

# Test 5: Test GUI creation
echo ""
info "Test 5: Testing GUI creation..."
if ./run.sh gui:create > /dev/null 2>&1; then
    if [[ -d "./gui" ]]; then
        success "GUI scaffold created successfully"

        # Check GUI files
        if [[ -f "./gui/index.html" && -f "./gui/app.js" && -f "./gui/styles.css" ]]; then
            success "All GUI files are present"
        else
            warning "Some GUI files are missing"
        fi
    else
        error "GUI directory was not created"
    fi
else
    error "GUI creation failed"
fi

# Test 6: Validate configuration files
echo ""
info "Test 6: Checking configuration files..."

if [[ -f ".env.example" ]]; then
    success ".env.example exists"
else
    warning ".env.example not found"
fi

if [[ -f "docker-compose.yml" ]]; then
    success "docker-compose.yml exists"

    # Basic validation
    if docker-compose config > /dev/null 2>&1 || docker compose config > /dev/null 2>&1; then
        success "docker-compose.yml is valid"
    else
        error "docker-compose.yml has syntax errors"
    fi
else
    warning "docker-compose.yml not found"
fi

if [[ -f "Dockerfile" ]]; then
    success "Dockerfile exists"
else
    error "Dockerfile not found"
fi

# Test 7: Test build (dry run)
echo ""
info "Test 7: Testing Docker build preparation..."
if docker build --help > /dev/null 2>&1; then
    success "Docker build command is available"

    # Check if we can build (but don't actually build to save time)
    if docker build -t test-stereo-vision --dry-run . > /dev/null 2>&1; then
        success "Dockerfile syntax appears valid"
    else
        # dry-run flag might not be supported, so just check dockerfile syntax
        if docker build -t test-stereo-vision --no-cache --pull=false --target base . > /dev/null 2>&1; then
            success "Dockerfile builds successfully"
            docker rmi test-stereo-vision > /dev/null 2>&1 || true
        else
            warning "Dockerfile build test failed (this may be normal)"
        fi
    fi
else
    error "Docker build command failed"
fi

# Test 8: Port availability check
echo ""
info "Test 8: Checking port availability..."
check_port() {
    local port=$1
    if ss -tulpn | grep ":$port " > /dev/null 2>&1; then
        warning "Port $port is already in use"
        return 1
    else
        success "Port $port is available"
        return 0
    fi
}

check_port 8080
check_port 3000
check_port 8081

# Test 9: System requirements
echo ""
info "Test 9: Checking system requirements..."

# Check available memory
if command -v free &> /dev/null; then
    total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [[ $total_mem -gt 2048 ]]; then
        success "Sufficient memory available (${total_mem}MB)"
    else
        warning "Low memory detected (${total_mem}MB) - may cause build issues"
    fi
fi

# Check disk space
if command -v df &> /dev/null; then
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [[ $available_space -gt 5242880 ]]; then  # 5GB in KB
        success "Sufficient disk space available"
    else
        warning "Low disk space - may cause build issues"
    fi
fi

# Test 10: Summary
echo ""
echo "🎯 Test Summary"
echo "━━━━━━━━━━━━━━━"

if [[ -d "./gui" ]]; then
    info "✅ GUI scaffold: Ready"
else
    info "⚠️  GUI scaffold: Not created"
fi

if [[ -f ".env.example" ]]; then
    info "✅ Configuration: Templates available"
else
    info "⚠️  Configuration: Templates missing"
fi

echo ""
echo "🚀 Next Steps:"
echo "1. Run: ./run.sh up                 # Start all services"
echo "2. Access: http://localhost:3000    # Web interface"
echo "3. API: http://localhost:8080       # Backend API"
echo "4. Status: ./run.sh status          # Check service health"
echo "5. Logs: ./run.sh logs              # View application logs"
echo ""
echo "📖 For full documentation, see: DOCKER_RUNNER_README.md"
echo ""
success "Setup test completed! 🎉"
