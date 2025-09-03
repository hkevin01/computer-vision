# Project Structure Cleanup Summary

## 🎯 Cleanup Completed

### Files Moved to `docs/setup/`

-- `DOCKER_SETUP.md` → `docs/setup/docker-setup.md`
-- `DOCKER_README.md` → `docs/setup/docker-readme.md`

### Files Moved to `docs/planning/`

-- `AI_ML_IMPROVEMENTS_SUMMARY.md` → `docs/planning/AI_ML_IMPROVEMENTS_SUMMARY.md`
-- `IMPLEMENTATION_PLAN.md` → `docs/architectural/IMPLEMENTATION_PLAN.md`
-- `IMPROVEMENTS_ROADMAP.md` → `docs/planning/IMPROVEMENTS_ROADMAP.md`

- `OPENCV_OPTIMIZATION.md` → `docs/planning/opencv-optimization.md`
- `PROJECT_MODERNIZATION_STRATEGY.md` → `docs/planning/modernization-strategy.md`

### Files Moved to `docs/process/`

- `DIRECTORY_CLEANUP_SUMMARY.md` → `docs/process/directory-cleanup.md`
- `WORKFLOW.md` → `docs/process/workflow.md`

### 🔗 Symbolic Links Created (for convenience)

- `DOCKER_SETUP.md` → `docs/setup/docker-setup.md`
- `QUICK_START.md` → `docs/setup/docker-readme.md`

### 📁 Root Directory Now Contains

**Essential Build Files:**

- `CMakeLists.txt`
- `run.sh`
- `launch_gui.sh`

**Docker Files:**

- `Dockerfile`
- `docker-compose.yml`
- `.env.example`
- `scripts/docker/docker-demo.sh`

**Project Files:**

- `README.md`
- `LICENSE`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`

**Build/Development:**

- `build/` (build artifacts)
- `src/` (source code)
- `include/` (headers)
- `tests/` (test files)
- `data/` (sample data)

**Organization:**

- `docs/` (all documentation)
- `scripts/` (utility scripts)
- `tools/` (development tools)

### 🚀 Quick Access Links

- `./run.sh` - Main build/run script
- `./docs/setup/docker-readme.md` - Docker usage guide
- `./docs/setup/docker-setup.md` - Docker setup overview
- `./README.md` - Project overview

## ✅ Benefits Achieved

1. **Clean Root**: Only essential files in project root
2. **Organized Docs**: All documentation properly categorized
3. **Quick Access**: Symlinks maintain easy access to key files
4. **Docker Ready**: Docker files remain in root for easy use
5. **Build Ready**: All build files (CMakeLists.txt, run.sh) in root

The project root is now clean and organized while maintaining full functionality!

### Remaining Tasks

- [ ] Complete repo-wide filename-only replacements for all moved docs and scripts (finish linking in `docs/` and `scripts/legacy/`).
- [x] Move docker-related scripts into `scripts/docker/` and leave shims in `scripts/legacy/`.
- [x] Add `add_subdirectory(test_programs)` to top-level `CMakeLists.txt` so tests build with the main project.
- [x] Add smoke/diagnostic scripts and initial test program scaffolding (`scripts/smoke.sh`, `scripts/diagnose_env.sh`, `test_programs/`).
- [ ] Populate `config/models_urls.sh` with real model URLs and SHA256 checksums (HITNet, RAFT-Stereo, CREStereo).
- [ ] Make new scripts executable locally: `chmod +x scripts/*.sh scripts/docker/*.sh build.sh`.
- [ ] Finish subdividing remaining `scripts/legacy/` into `scripts/reorg/` and `scripts/debug/` and update shims.
- [ ] Add deterministic sample data to `data/stereo_images/` and `data/calibration/` for smoke tests.
- [ ] Update CI to avoid compiling ONNX C++ tests or install ONNX Runtime C++ dev packages, and re-run CI.
- [ ] Run a focused markdown lint/format pass on edited docs to clean remaining warnings.
