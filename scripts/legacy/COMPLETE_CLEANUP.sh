# Symlink Creation Script

## To complete the cleanup, run these commands:

```bash
# Navigate to project root
cd /home/kevin/Projects/computer-vision

# Create convenient symlinks for Docker documentation
ln -sf docs/setup/docker-setup.md DOCKER_SETUP.md
ln -sf docs/setup/docker-readme.md QUICK_START.md

# Remove empty planning files in root (they exist in documentation/)
rm -f docs/planning/AI_ML_IMPROVEMENTS_SUMMARY.md
rm -f docs/architectural/IMPLEMENTATION_PLAN.md
rm -f IMPROVEMENTS_ROADMAP.md
rm -f OPENCV_OPTIMIZATION.md
rm -f PROJECT_MODERNIZATION_STRATEGY.md
rm -f DIRECTORY_CLEANUP_SUMMARY.md
rm -f WORKFLOW.md

# Verify the cleanup
echo "Root files after cleanup:"
ls -la *.md *.yml *.yaml *.sh *.txt 2>/dev/null | head -20
```

## Manual Verification Commands:

```bash
# Check that symlinks work
ls -la DOCKER_SETUP.md QUICK_START.md

# Verify Docker files are accessible
cat DOCKER_SETUP.md | head -5

# Test run.sh still works
./run.sh --help

# Check documentation organization
ls -la docs/
```
