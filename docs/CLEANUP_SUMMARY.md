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

#### Clean-up and Organization

- [x] Complete repo-wide filename-only replacements for all moved docs and scripts (finish linking in `docs/` and `scripts/legacy/`).
- [x] Move docker-related scripts into `scripts/docker/` and leave shims in `scripts/legacy/`.
- [x] Add `add_subdirectory(test_programs)` to top-level `CMakeLists.txt` so tests build with the main project.
- [x] Add smoke/diagnostic scripts and initial test program scaffolding (`scripts/smoke.sh`, `scripts/diagnose_env.sh`, `test_programs/`).
- [ ] Make new scripts executable locally: `chmod +x scripts/*.sh scripts/docker/*.sh build.sh`.
- [ ] Finish subdividing remaining `scripts/legacy/` into `scripts/reorg/` and `scripts/debug/` and update shims.
- [ ] Run a focused markdown lint/format pass on edited docs to clean remaining warnings.

#### Build System and Development Experience

- [x] Create `CMakePresets.json` with presets: cpu-debug, cuda-release, hip-release, cpu-onnx, cuda-onnx-trt
- [x] Add `.vscode/tasks.json` and `.vscode/launch.json` for VS Code integration
- [x] Implement generated config header (`include/config/config_features.hpp`) from CMake with compile-time toggles
- [x] Replace `config/models_urls.sh` with structured `config/models.yaml` (HITNet, RAFT-Stereo, CREStereo with checksums)
- [ ] Add CI pipelines for Linux/Windows/macOS with CUDA/HIP/CPU-only profiles and dependency caching

#### AI/ML Infrastructure

- [ ] Create `ModelRegistry` class that loads `config/models.yaml` with SHA256 validation and provider preferences
- [x] Replace `config/models_urls.sh` with structured `config/models.yaml` (HITNet, RAFT-Stereo, CREStereo with checksums)
- [ ] Implement TensorRT engine caching under `data/models/cache/` with precision settings and safe fallback
- [ ] Add ONNX Runtime provider selection and session optimization based on available backends

#### Testing and Quality

- [ ] Add unit tests: ai_model_registry_test, disparity_reprojection_test, backend_selection_test
- [ ] Add deterministic sample data to `data/stereo_images/` and `data/calibration/` for smoke tests
- [ ] Update CI to avoid compiling ONNX C++ tests or install ONNX Runtime C++ dev packages, and re-run CI
- [ ] Add integration tests with golden outputs for CPU/CUDA/HIP backends with tolerance checks

#### Performance and Benchmarking

- [ ] Create benchmark CLI (`src/tools/benchmark_app.cpp`) with CSV/JSON output to `reports/benchmarks/`
- [ ] Add structured logging with spdlog and JSON sink to `logs/` with session UUID and performance metrics
- [ ] Implement streaming pipeline with double/triple buffering and CUDA/HIP stream overlap

#### Documentation and UX

- [ ] Consolidate overlapping docs into `docs/` with mkdocs or Docusaurus site structure
- [ ] Add `docs/SETUP_REQUIREMENTS.md` with dependency matrix and GPU driver troubleshooting FAQ
- [ ] Implement GUI parameter persistence and first-run wizard for backend detection and model setup
- [ ] Add sample data licensing and downloadable stereo pairs under `data/stereo_images/`
