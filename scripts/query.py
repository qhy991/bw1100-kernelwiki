#!/usr/bin/env python3
"""
Query the ROCm-KernelWiki-Q knowledge base.

Usage:
    python3 scripts/query.py "how to optimize GEMM on MI300X"
    python3 scripts/query.py --tag mfma --type kernel
    python3 scripts/query.py --technique bank-conflict-padding --architecture cdna3
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = ROOT / "wiki"
SOURCES_DIR = ROOT / "sources"

# Friendly short names for --type (docs use these), mapped to canonical page types.
TYPE_ALIASES = {
    "hardware": "wiki-hardware", "technique": "wiki-technique", "kernel": "wiki-kernel",
    "pattern": "wiki-pattern", "language": "wiki-language", "migration": "wiki-migration",
    "experiment": "source-experiment", "pr": "source-pr", "doc": "source-doc", "blog": "source-blog",
}

# Tags that saturate the corpus and carry no discriminative signal in ranking.
STOP_TAGS = {"rocm", "rocm-kernel"}


def extract_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None, content
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None, content
    return fm, content


def collect_pages():
    """Collect all pages with frontmatter and content."""
    pages = []
    for dir_path in [WIKI_DIR, SOURCES_DIR]:
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.rglob("*.md")):
            fm, content = extract_frontmatter(md_file)
            if fm:
                rel = md_file.relative_to(ROOT)
                pages.append({"path": str(rel), "content": content, **fm})
    return pages


def search_keyword(pages, query):
    """Simple keyword search across title, tags, and content."""
    query_lower = query.lower()
    query_terms = query_lower.split()

    scored = []
    for p in pages:
        score = 0
        title = p.get("title", "").lower()
        tags = [t.lower() for t in p.get("tags", [])]
        content_lower = p.get("content", "").lower()
        archs = [a.lower() for a in p.get("architectures", [])]

        # Title match (highest weight)
        for term in query_terms:
            if term in title:
                score += 10

        # Tag match (ignore corpus-saturating stop tags)
        for term in query_terms:
            for tag in tags:
                if tag in STOP_TAGS:
                    continue
                if term in tag:
                    score += 5

        # Architecture match
        for term in query_terms:
            for arch in archs:
                if term in arch:
                    score += 3

        # Content match
        for term in query_terms:
            if term in content_lower:
                score += 1

        # Prefer curated wiki pages over raw PR/source stubs when otherwise tied.
        if score > 0 and str(p.get("type", "")).startswith("wiki-"):
            score += 4

        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored]


def filter_pages(pages, **filters):
    """Filter pages by structured criteria."""
    result = pages
    if filters.get("tag"):
        tags = filters["tag"] if isinstance(filters["tag"], list) else [filters["tag"]]
        result = [p for p in result if any(t in p.get("tags", []) for t in tags)]
    if filters.get("type"):
        want = TYPE_ALIASES.get(filters["type"], filters["type"])
        result = [p for p in result if p.get("type") == want]
    if filters.get("architecture"):
        archs = filters["architecture"] if isinstance(filters["architecture"], list) else [filters["architecture"]]
        result = [p for p in result if any(a in p.get("architectures", []) for a in archs)]
    if filters.get("technique"):
        techs = filters["technique"] if isinstance(filters["technique"], list) else [filters["technique"]]
        result = [p for p in result if any(t in p.get("techniques", p.get("tags", [])) for t in techs)]
    if filters.get("kernel_type"):
        kts = filters["kernel_type"] if isinstance(filters["kernel_type"], list) else [filters["kernel_type"]]
        result = [p for p in result if any(kt in p.get("kernel_types", []) for kt in kts)]
    if filters.get("operator"):
        operators = filters["operator"] if isinstance(filters["operator"], list) else [filters["operator"]]
        operator_set = {str(op).lower() for op in operators}
        result = [
            p for p in result
            if operator_set & {
                str(value).lower()
                for field in ("kernel_types", "tags")
                for value in p.get(field, [])
            }
        ]
    if filters.get("confidence"):
        result = [p for p in result if p.get("confidence") == filters["confidence"]]
    return result


def format_results(pages, limit=10, compact=False):
    """Format search results for display."""
    if not pages:
        return "No results found."

    lines = []
    for p in pages[:limit]:
        eid = p.get("id", "?")
        title = p.get("title", eid)
        ptype = p.get("type", "?")

        if compact:
            # One-line-per-result, matching the cross-KernelWiki compact contract.
            lines.append(f"  [{ptype}] {eid}: {title}  ({p['path']})")
            continue

        conf = p.get("confidence", "?")
        archs = ", ".join(p.get("architectures", []))
        tags = ", ".join(p.get("tags", [])[:5])
        desc = p.get("description", "")[:80]

        lines.append(f"## {title}")
        lines.append(f"  ID:    {eid}")
        lines.append(f"  Type:  {ptype}")
        lines.append(f"  Arch:  {archs}")
        lines.append(f"  Conf:  {conf}")
        if desc:
            lines.append(f"  Desc:  {desc}")
        if tags:
            lines.append(f"  Tags:  {tags}")
        lines.append(f"  Path:  {p['path']}")
        lines.append("")

    if len(pages) > limit:
        lines.append(f"... and {len(pages) - limit} more results")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query ROCm-KernelWiki-Q")
    parser.add_argument("query", nargs="?", help="Natural language search query")
    parser.add_argument("--tag", "-t", action="append", help="Filter by tag")
    parser.add_argument("--type", help="Filter by page type (wiki-hardware, wiki-technique, etc.)")
    parser.add_argument("--architecture", "-a", action="append", help="Filter by architecture (cdna3, cdna4)")
    parser.add_argument("--technique", action="append", help="Filter by technique")
    parser.add_argument("--kernel-type", action="append", help="Filter by kernel type")
    parser.add_argument("--operator", action="append", help="Common cross-KernelWiki operator filter")
    parser.add_argument("--confidence", help="Filter by confidence level")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    parser.add_argument("--compact", action="store_true", help="Compact one-line-per-result output")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths, one per line")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show content snippets")

    args = parser.parse_args()

    pages = collect_pages()

    # Apply structured filters
    if args.tag or args.type or args.architecture or args.technique or args.kernel_type or args.operator or args.confidence:
        results = filter_pages(
            pages,
            tag=args.tag,
            type=args.type,
            architecture=args.architecture,
            technique=args.technique,
            kernel_type=args.kernel_type,
            operator=args.operator,
            confidence=args.confidence,
        )
        if args.query:
            results = search_keyword(results, args.query)
    elif args.query:
        results = search_keyword(pages, args.query)
    else:
        parser.print_help()
        return 1

    if args.paths_only:
        for p in results[:args.limit]:
            print(p["path"])
        return 0

    print(format_results(results, args.limit, compact=args.compact))

    # Show snippets if verbose
    if args.verbose and results:
        for p in results[:3]:
            lines = p.get("content", "").split("\n")
            # Show first 20 lines after frontmatter
            in_fm = False
            shown = 0
            print(f"\n--- {p.get('id', '?')} snippet ---")
            for line in lines:
                if line.strip() == "---":
                    in_fm = not in_fm
                    continue
                if not in_fm:
                    print(line)
                    shown += 1
                    if shown >= 20:
                        break

    return 0


if __name__ == "__main__":
    sys.exit(main())
