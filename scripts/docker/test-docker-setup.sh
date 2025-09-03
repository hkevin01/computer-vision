#!/usr/bin/env bash
# Smoke test for docker setup (moved from scripts/legacy)
echo "Running quick docker smoke tests..."
docker-compose config >/dev/null 2>&1 && echo "docker-compose config OK" || echo "docker-compose config failed"
exit 0
