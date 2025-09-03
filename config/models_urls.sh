#!/usr/bin/env bash
# Shell-based mapping of model download URLs and optional sha256 checksums.
# Populate the URL and SHA256 for each model you want the fetch script to download.
# Example:
# HITNET_URL="https://raw.githubusercontent.com/your/repo/main/models/hitnet.onnx"
# HITNET_SHA="abcd1234..."

# HITNet
HITNET_URL=""
HITNET_SHA=""

# RAFT-Stereo
RAFT_STEREO_URL=""
RAFT_STEREO_SHA=""

# CREStereo
CRESTEREO_URL=""
CRESTEREO_SHA=""

export HITNET_URL HITNET_SHA RAFT_STEREO_URL RAFT_STEREO_SHA CRESTEREO_URL CRESTEREO_SHA
