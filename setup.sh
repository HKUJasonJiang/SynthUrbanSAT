#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
SynthUrbanSAT setup dispatcher

Usage:
  bash setup.sh train [args...]       Run train_pipeline/setup.sh
  bash setup.sh osm [args...]         Install/check osm_pipeline dependencies
  bash setup.sh generation [args...]  Run generation_pipeline/setup.sh
  bash setup.sh all                   Run train, osm, then generation with defaults

Each pipeline remains independently runnable from its own directory.
EOF
}

run_pipeline() {
    local name="$1"
    shift
    case "$name" in
        train)
            cd "$ROOT/train_pipeline"
            bash setup.sh "$@"
            ;;
        osm)
            cd "$ROOT/osm_pipeline"
            bash setup.sh "$@"
            ;;
        generation)
            cd "$ROOT/generation_pipeline"
            bash setup.sh "$@"
            ;;
        all)
            "$ROOT/setup.sh" train
            "$ROOT/setup.sh" osm
            "$ROOT/setup.sh" generation
            ;;
        -h|--help|help|"")
            usage
            ;;
        *)
            echo "Unknown setup target: $name" >&2
            usage >&2
            exit 2
            ;;
    esac
}

run_pipeline "${1:-help}" "${@:2}"
