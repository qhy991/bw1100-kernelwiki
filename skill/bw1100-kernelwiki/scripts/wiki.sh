#!/usr/bin/env bash
set -euo pipefail
skill=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
exec bash "$skill/knowledge/bwiki" "$@"
