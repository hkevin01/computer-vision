#!/bin/bash
echo "🔄 Updating Docker-first setup files..."

# Backup original files
if [[ -f "run.sh" ]]; then
    cp run.sh run.sh.backup
    echo "✅ Backed up original run.sh to run.sh.backup"
fi

if [[ -f "docker-compose.yml" ]]; then
    cp docker-compose.yml docker-compose.yml.backup
    echo "✅ Backed up original docker-compose.yml"
fi

# Copy new files
if [[ -f "run.sh.new" ]]; then
    cp run.sh.new run.sh
    chmod +x run.sh
    echo "✅ Updated run.sh with enhanced Docker-first version"
fi

if [[ -f "docker/docker-compose.yml.new" ]]; then
    cp docker/docker-compose.yml.new docker-compose.yml
    echo "✅ Updated docker-compose.yml with new service configuration (from docker/docker-compose.yml.new)"
fi

# Make test script executable
if [[ -f "test-docker-setup.sh" ]]; then
    chmod +x test-docker-setup.sh
    echo "✅ Made test-docker-setup.sh executable"
fi

echo ""
echo "🎉 Setup update complete!"
echo ""
echo "🚀 Quick start:"
echo "  scripts/docker/test-docker-setup.sh     # Test the setup"
echo "  ./run.sh gui:create        # Create web GUI"
echo "  ./run.sh up                # Start all services"
echo "  ./run.sh gui:open          # Open web interface"
echo ""
echo "📖 See DOCKER_RUNNER_README.md for full documentation"
