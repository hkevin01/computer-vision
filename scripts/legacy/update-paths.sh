#!/bin/bash
set -e

echo "🔧 POST-REORGANIZATION CONFIGURATION UPDATE"
echo "=========================================="
echo "Updating paths and references after reorganization"
echo ""

cd /home/kevin/Projects/computer-vision

# === PHASE 1: Update Docker Compose Configuration ===
echo "🐳 PHASE 1: Updating Docker Compose Configuration"
echo "------------------------------------------------"

if [ -f "docker/docker-compose.yml" ]; then
    # Update Dockerfile path in docker-compose.yml
    sed -i 's|dockerfile: Dockerfile|dockerfile: ../Dockerfile|g' docker/docker-compose.yml

    # Update context path if needed
    sed -i 's|context: \.|context: ..|g' docker/docker-compose.yml

    # Update GUI context path
    sed -i 's|context: \${GUI_PATH:-\./gui}|context: ../gui|g' docker/docker-compose.yml

    echo "✅ Updated: docker/docker-compose.yml paths"
else
    echo "⚠️  docker/docker-compose.yml not found"
fi

# === PHASE 2: Update Main Dockerfile ===
echo ""
echo "📦 PHASE 2: Updating Main Dockerfile"
echo "-----------------------------------"

if [ -f "Dockerfile" ]; then
    # No changes needed to main Dockerfile as it uses relative paths
    echo "✅ Main Dockerfile paths are relative (no changes needed)"
else
    echo "⚠️  Main Dockerfile not found"
fi

# === PHASE 3: Update GUI Dockerfile ===
echo ""
echo "🌐 PHASE 3: Updating GUI Dockerfile"
echo "----------------------------------"

if [ -f "gui/Dockerfile" ]; then
    # GUI Dockerfile uses relative paths, should be fine
    echo "✅ GUI Dockerfile paths are relative (no changes needed)"
else
    echo "⚠️  gui/Dockerfile not found"
fi

# === PHASE 4: Update Run Script ===
echo ""
echo "🔧 PHASE 4: Updating Run Script"
echo "------------------------------"

if [ -f "run.sh" ]; then
    # Update docker-compose.yml path in run.sh
    cp run.sh run.sh.backup-reorg
    sed -i 's|docker-compose\.yml|docker/docker-compose.yml|g' run.sh
    sed -i 's|COMPOSE_FILE="[^"]*"|COMPOSE_FILE="docker/docker-compose.yml"|g' run.sh
    echo "✅ Updated: run.sh docker-compose paths"
else
    echo "⚠️  run.sh not found"
fi

# === PHASE 5: Create New Enhanced Run Script ===
echo ""
echo "🚀 PHASE 5: Creating Enhanced Run Script"
echo "---------------------------------------"

cat > run-clean.sh << 'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

# === Enhanced Docker-first Stereo Vision Application Runner ===
# Updated for clean project structure

# Set error trap
trap 'echo "❌ Error occurred at line $LINENO. Exit code: $?" >&2' ERR

# === Configuration Defaults ===
COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-.env}"
GUI_PATH="${GUI_PATH:-./gui}"

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

# === Docker Functions ===
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
    print_status "Using compose file: $COMPOSE_FILE"
}

# === Command Functions ===
cmd_up() {
    print_header
    check_docker
    detect_compose

    print_status "Starting all services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d

    print_success "Services started successfully!"
    print_status "Access points:"
    print_status "  🌐 Web GUI: http://localhost:3000"
    print_status "  🔗 API: http://localhost:8080"
}

cmd_down() {
    print_header
    detect_compose

    print_status "Stopping all services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" down

    print_success "Services stopped successfully!"
}

cmd_build() {
    print_header
    check_docker
    detect_compose

    print_status "Building all services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build "$@"

    print_success "Build completed!"
}

cmd_logs() {
    local service="${1:-}"
    detect_compose

    if [ -n "$service" ]; then
        print_status "Showing logs for service: $service"
        $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f "$service"
    else
        print_status "Showing logs for all services"
        $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f
    fi
}

cmd_status() {
    print_header
    detect_compose

    print_status "Application Status:"
    echo ""

    # Container status
    echo "📦 Containers:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
    echo ""

    # Network status
    echo "🌐 Services:"
    if curl -s "http://localhost:3000/health" > /dev/null; then
        echo "  ✅ GUI: http://localhost:3000"
    else
        echo "  ❌ GUI: http://localhost:3000 (not responding)"
    fi

    if curl -s "http://localhost:8080/health" > /dev/null; then
        echo "  ✅ API: http://localhost:8080"
    else
        echo "  ❌ API: http://localhost:8080 (not responding)"
    fi
}

cmd_gui_open() {
    print_status "Opening web interface..."

    local gui_url="http://localhost:3000"

    if command -v xdg-open &> /dev/null; then
        xdg-open "$gui_url"
    elif command -v open &> /dev/null; then
        open "$gui_url"
    else
        print_status "Please manually open: $gui_url"
    fi
}

cmd_help() {
    print_header
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  up                Start all services"
    echo "  down              Stop all services"
    echo "  build             Build all services"
    echo "  logs [SERVICE]    Show logs (optional service name)"
    echo "  status            Show service status"
    echo "  gui:open          Open web interface"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 up             # Start all services"
    echo "  $0 logs api       # Show API logs"
    echo "  $0 gui:open       # Open web GUI"
    echo ""
    echo "Configuration:"
    echo "  Compose file: $COMPOSE_FILE"
    echo "  Env file: $ENV_FILE"
    echo "  GUI path: $GUI_PATH"
}

# === Main Execution ===
main() {
    if [[ $# -eq 0 ]]; then
        cmd_help
        exit 0
    fi

    local command="$1"
    shift || true

    case "$command" in
        up)
            cmd_up "$@"
            ;;
        down)
            cmd_down "$@"
            ;;
        build)
            cmd_build "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        gui:open)
            cmd_gui_open "$@"
            ;;
        help|--help|-h)
            cmd_help
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
EOF

chmod +x run-clean.sh
echo "✅ Created: run-clean.sh (updated for clean structure)"

# === PHASE 6: Create Docker Directory Scripts ===
echo ""
echo "📦 PHASE 6: Creating Docker Directory Scripts"
echo "--------------------------------------------"

# Create docker/run-docker.sh
cat > docker/run-docker.sh << 'EOF'
#!/bin/bash
# Direct Docker operations from docker directory

COMPOSE_FILE="docker-compose.yml"

case "$1" in
    up)
        docker-compose -f "$COMPOSE_FILE" up -d
        ;;
    down)
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    build)
        docker-compose -f "$COMPOSE_FILE" build
        ;;
    logs)
        docker-compose -f "$COMPOSE_FILE" logs -f ${2:-}
        ;;
    *)
        echo "Usage: $0 {up|down|build|logs [service]}"
        exit 1
        ;;
esac
EOF

chmod +x docker/run-docker.sh
echo "✅ Created: docker/run-docker.sh"

# === PHASE 7: Update .gitignore ===
echo ""
echo "📝 PHASE 7: Updating .gitignore"
echo "------------------------------"

if [ -f ".gitignore" ]; then
    # Add entries for new structure if not already present
    echo "" >> .gitignore
    echo "# Project reorganization" >> .gitignore
    echo "run.sh.backup-reorg" >> .gitignore
    echo "*.backup-reorg" >> .gitignore
    echo "✅ Updated: .gitignore"
else
    echo "⚠️  .gitignore not found"
fi

# === PHASE 8: Create Quick Reference ===
echo ""
echo "📖 PHASE 8: Creating Quick Reference"
echo "-----------------------------------"

cat > QUICK_START.md << 'EOF'
# Quick Start Guide - Clean Structure

## Project Structure
```
computer-vision/
├── docker/          # All Docker files
├── scripts/         # All shell scripts
├── docs/           # All documentation
├── config/         # Configuration files
├── api/            # API server
├── web/            # Web assets
├── gui/            # Web GUI
└── run.sh          # Main entry point
```

## Quick Commands
```bash
# Start services
./run.sh up

# Check status
./run.sh status

# View logs
./run.sh logs

# Open web GUI
./run.sh gui:open

# Stop services
./run.sh down
```

## Alternative (from docker directory)
```bash
cd docker/
./run-docker.sh up
```
EOF

echo "✅ Created: QUICK_START.md"

echo ""
echo "🎉 CONFIGURATION UPDATE COMPLETE!"
echo "================================"
echo ""
echo "✅ Updated paths and references for clean structure"
echo "✅ Created enhanced run scripts"
echo "✅ Added directory-specific utilities"
echo "✅ Updated configuration files"
echo ""
echo "🚀 Ready to use reorganized project!"
