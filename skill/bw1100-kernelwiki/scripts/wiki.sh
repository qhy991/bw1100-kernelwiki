#!/usr/bin/env bash
set -euo pipefail
skill=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f "$skill/knowledge/bwiki" ]]; then root="$skill/knowledge";
elif [[ -f "$skill/../../bwiki" ]]; then root="$skill/../..";
else printf 'The canonical BW1100 knowledge link is missing.
' >&2; exit 1; fi
exec bash "$root/bwiki" "$@"
