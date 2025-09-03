#!/bin/bash
# Legacy shim: forward to scripts/docker/update-docker-setup.sh if present
NEW="$(dirname "$0")/../docker/update-docker-setup.sh"
if [ -x "$NEW" ]; then
    exec "$NEW" "$@"
else
    echo "update-docker-setup.sh not found in new location: $NEW"
    echo "Falling back to legacy behavior."
    # legacy behavior placeholder
    exit 2
fi
    if [[ -f "run.sh" ]]; then
        echo "✅ run.sh exists"
    fi
    if [[ -f "docker-compose.yml" ]]; then
        echo "✅ docker-compose.yml exists"
    fi
fi
