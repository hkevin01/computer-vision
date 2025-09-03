# Repository Move Map

This file lists files moved during the repository reorganization and their new locations.

Old path -> New path

- DOCKER_SETUP.md -> docs/setup/docker-setup.md
- DOCKER_README.md -> docs/setup/docker-readme.md
- AI_ML_IMPROVEMENTS_SUMMARY.md -> docs/planning/AI_ML_IMPROVEMENTS_SUMMARY.md
- IMPLEMENTATION_PLAN.md -> docs/architectural/IMPLEMENTATION_PLAN.md
- IMPROVEMENTS_ROADMAP.md -> docs/planning/IMPROVEMENTS_ROADMAP.md
- README_CLEANUP.md -> docs/guides/README_CLEANUP.md

Scripts reorganized:

- scripts/legacy/* -> subdivided into scripts/docker/, scripts/reorg/, scripts/debug/, with symlinks left in scripts/legacy/ for compatibility.

Docker scripts moved:

- docker-demo.sh -> scripts/docker/docker-demo.sh
- update-docker-setup.sh -> scripts/docker/update-docker-setup.sh
- test-docker-setup.sh -> scripts/docker/test-docker-setup.sh

If you cannot find a file, try searching the `docs/` and `scripts/` directories.

Recent small moves:

- run.sh.new (backup) kept in root as run.sh.new -> (no move; kept as backup)
- docker-compose.yml.new -> docker/docker-compose.yml.new
- test-connection.html -> docs/moved_files/test-connection.html
- test_args.cpp -> test_programs/test_args.cpp
