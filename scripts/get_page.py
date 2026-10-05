#!/usr/bin/env python3
"""
Fetch a single page by its `id` (deterministic lookup, the basic retrieval primitive).

Usage:
    python3 scripts/get_page.py hw-mfma-matrix-core                      # full page
    python3 scripts/get_page.py pr-composable_kernel-3540                # a PR source page
    python3 scripts/get_page.py hw-gws --frontmatter                     # frontmatter only
    python3 scripts/get_page.py hw-gws --field sources                   # one field's value
    python3 scripts/get_page.py kernel-flash-attention-rocm --body-only  # markdown body only
    python3 scripts/get_page.py mfma                                     # fuzzy: suggest ids
"""

import argparse
import re
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = [ROOT / "wiki", ROOT / "sources"]


def extract_frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    try:
        return (yaml.safe_load(m.group(1)) or {}), m.group(2)
    except yaml.YAMLError:
        return {}, text


def build_index():
    index = {}
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for md in d.rglob("*.md"):
            # encoding="utf-8" keeps reads correct on Windows (KerSor RAG retrieval).
            fm, body = extract_frontmatter(md.read_text(encoding="utf-8"))
            pid = fm.get("id")
            if pid:
                index[pid] = (md, fm, body)
    return index


def main():
    ap = argparse.ArgumentParser(description="Fetch a page by id")
    ap.add_argument("id", help="page id (e.g. hw-mfma-matrix-core)")
    ap.add_argument("--frontmatter", "--frontmatter-only", dest="frontmatter", action="store_true", help="print frontmatter only")
    ap.add_argument("--field", help="print just one frontmatter field")
    ap.add_argument(
        "--body-only",
        action="store_true",
        help="print markdown body without YAML frontmatter",
    )
    args = ap.parse_args()

    index = build_index()

    if args.id not in index:
        # fuzzy: substring match on id
        cands = sorted(k for k in index if args.id.lower() in k.lower())
        if not cands:
            print(f"No page with id {args.id!r}. (Tried {len(index)} pages.)", file=sys.stderr)
            return 1
        print(f"No exact id {args.id!r}. Did you mean:", file=sys.stderr)
        for c in cands[:15]:
            print(f"  {c}  ->  {index[c][0].relative_to(ROOT)}", file=sys.stderr)
        return 1

    path, fm, body = index[args.id]

    if args.field:
        print(fm.get(args.field, ""))
        return 0
    if args.frontmatter:
        print(yaml.safe_dump(fm, sort_keys=False).strip())
        return 0
    if args.body_only:
        print(body.strip())
        return 0

    print(f"# Path: {path.relative_to(ROOT)}\n")
    print(yaml.safe_dump(fm, sort_keys=False).strip())
    print("\n---\n")
    print(body.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
