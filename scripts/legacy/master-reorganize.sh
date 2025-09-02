#!/bin/bash
set -e

echo "🚀 MASTER PROJECT REORGANIZATION SCRIPT"
echo "======================================="
echo "Complete transformation to clean, professional structure"
echo ""

cd /home/kevin/Projects/computer-vision

# Make sure all scripts are executable
chmod +x *.sh 2>/dev/null || true

# === STEP 1: Create Backup ===
echo "🔒 STEP 1: Creating Safety Backup"
echo "--------------------------------"

backup_dir="../computer-vision-backup-$(date +%Y%m%d-%H%M%S)"
if [ ! -d "$backup_dir" ]; then
    cp -r . "$backup_dir"
    echo "✅ Backup created: $backup_dir"
else
    echo "✅ Backup already exists"
fi

# === STEP 2: Run Main Reorganization ===
echo ""
echo "🧹 STEP 2: Running Main Reorganization"
echo "-------------------------------------"

if [ -f "reorganize-project.sh" ]; then
    ./reorganize-project.sh
    echo "✅ Main reorganization completed"
else
    echo "❌ reorganize-project.sh not found"
    exit 1
fi

# === STEP 3: Update Paths and Configuration ===
echo ""
echo "🔧 STEP 3: Updating Paths and Configuration"
echo "------------------------------------------"

if [ -f "update-paths.sh" ]; then
    ./update-paths.sh
    echo "✅ Path updates completed"
else
    echo "❌ update-paths.sh not found"
    exit 1
fi

# === STEP 4: Verify Organization ===
echo ""
echo "🔍 STEP 4: Verifying Organization"
echo "--------------------------------"

if [ -f "verify-organization.sh" ]; then
    ./verify-organization.sh
    echo "✅ Verification completed"
else
    echo "❌ verify-organization.sh not found"
    exit 1
fi

# === STEP 5: Final Setup ===
echo ""
echo "⚡ STEP 5: Final Setup and Testing"
echo "---------------------------------"

# Make run scripts executable
chmod +x run*.sh 2>/dev/null || true
chmod +x docker/*.sh 2>/dev/null || true

# Test basic functionality
echo "Testing main run script..."
if [ -f "run.sh" ]; then
    if ./run.sh help &>/dev/null; then
        echo "✅ run.sh is functional"
    else
        echo "⚠️  run.sh may have issues"
    fi
fi

# Test enhanced run script
echo "Testing enhanced run script..."
if [ -f "run-clean.sh" ]; then
    if ./run-clean.sh help &>/dev/null; then
        echo "✅ run-clean.sh is functional"
    else
        echo "⚠️  run-clean.sh may have issues"
    fi
fi

# === COMPLETION SUMMARY ===
echo ""
echo "🎉 COMPLETE PROJECT REORGANIZATION FINISHED!"
echo "============================================"
echo ""
echo "✨ Transformation Results:"
echo "  📦 Docker files → docker/"
echo "  ⚙️  Configuration → config/"
echo "  📚 Documentation → docs/"
echo "  🔧 Shell scripts → scripts/"
echo "  🌐 API server → api/"
echo "  🧪 Web assets → web/"
echo "  🛠️  Development tools → tools/"
echo ""
echo "🎯 Clean Professional Structure Achieved!"
echo "  ✅ Root directory decluttered"
echo "  ✅ Logical file organization"
echo "  ✅ Professional project layout"
echo "  ✅ Enhanced maintainability"
echo ""
echo "📍 Key Entry Points:"
echo "  ./run.sh              # Main application runner"
echo "  ./run-clean.sh        # Enhanced runner (clean paths)"
echo "  docker/               # All Docker configuration"
echo "  scripts/              # All utility scripts"
echo "  docs/                 # All documentation"
echo ""
echo "🚀 Ready to Deploy:"
echo "  ./run.sh up           # Start Docker services"
echo "  ./run.sh status       # Check service status"
echo "  ./run.sh gui:open     # Open web interface"
echo ""
echo "📖 Documentation:"
echo "  README.md             # Updated main documentation"
echo "  QUICK_START.md        # Quick reference guide"
echo "  docs/                 # Detailed documentation"
echo ""
echo "🔄 Rollback Available:"
echo "  Backup location: $backup_dir"
echo "  To rollback: rm -rf . && cp -r $backup_dir/* ."
echo ""
echo "✨ Your Docker-first computer vision project is now professionally organized!"
echo "🎯 Structure follows industry best practices for maintainability and scalability"
echo ""
echo "Next: Test your reorganized setup with './run.sh up'"
