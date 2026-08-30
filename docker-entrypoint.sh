#!/bin/sh
# Seed the runtime model cache from the image-baked copy, then hand off to
# the CMD (uvicorn).
#
# Why: the Dockerfile bakes ~1.3 GB of HF weights into $MODEL_CACHE_SEED at
# build time. docker-compose mounts a named volume at $MODEL_CACHE_DIR so
# the cache survives container restarts — but that volume starts empty and
# shadows anything the image put at that path. Without this seed step the
# app would re-download every weight on a fresh volume (slow, and broken
# with no network).
#
# The copy runs only when the runtime cache has no model snapshot yet, so
# restarts with a warm volume are instant.

set -e

SEED_DIR="${MODEL_CACHE_SEED:-/opt/model-seed}"
RUNTIME_DIR="${MODEL_CACHE_DIR:-/app/.model_cache}"

if [ -d "$SEED_DIR" ] && [ -z "$(find "$RUNTIME_DIR" -maxdepth 2 -name 'models--*' -print -quit 2>/dev/null)" ]; then
    echo "seeding model cache: $SEED_DIR -> $RUNTIME_DIR"
    mkdir -p "$RUNTIME_DIR"
    cp -a "$SEED_DIR"/. "$RUNTIME_DIR"/
    echo "model cache seeded."
else
    echo "model cache already populated at $RUNTIME_DIR; skipping seed."
fi

exec "$@"
