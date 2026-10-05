#!/usr/bin/env python3
"""
Generate cross-reference indices (queries/*.md) + a machine-readable manifest
(queries/pages.json, queries/INDEX.md) from page frontmatter.

Vocabulary is loaded from data/tags.yaml at runtime (no hardcoded lists), and
non-canonical tags are normalized through data/tags.yaml `aliases`.

Usage:
    python3 scripts/generate-indices.py
"""

import json
import re
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WIKI_DIR = ROOT / "wiki"
SOURCES_DIR = ROOT / "sources"
QUERIES_DIR = ROOT / "queries"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def extract_frontmatter(filepath):
    with open(filepath) as f:
        content = f.read()
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None


def collect_pages():
    pages = []
    for dir_path in [WIKI_DIR, SOURCES_DIR]:
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.rglob("*.md")):
            fm = extract_frontmatter(md_file)
            if fm:
                pages.append({"path": str(md_file.relative_to(ROOT)), **fm})
    return pages


def link(path):
    """Index files live in queries/, so links to wiki/sources need a ../ prefix."""
    return f"../{path}"


def dedupe(entries):
    seen, out = set(), []
    for e in entries:
        eid = e.get("id", e["path"])
        if eid not in seen:
            seen.add(eid)
            out.append(e)
    return sorted(out, key=lambda x: x.get("id", x["path"]))


def grouped_index(title, pages, field, normalize=None, valid=None, line_fn=None):
    """Generic 'group pages by each value in <field>' index."""
    groups = defaultdict(list)
    for p in pages:
        vals = p.get(field, [])
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            cv = normalize(v) if normalize else v
            if valid is not None and cv not in valid:
                continue
            groups[cv].append(p)

    lines = [f"# {title}\n"]
    for key in sorted(groups):
        entries = dedupe(groups[key])
        lines.append(f"\n## {key} ({len(entries)} pages)\n")
        for e in entries:
            lines.append(line_fn(e))
    return "\n".join(lines)


def main():
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    tags = load_yaml(DATA_DIR / "tags.yaml")
    aliases = tags.get("aliases", {}) or {}
    hw_vocab = set(tags.get("hardware_features", {}).keys())
    tech_vocab = set(tags.get("techniques", {}).keys())
    arch_vocab = {a["id"] for a in tags.get("architectures", [])}

    def norm(v):
        return aliases.get(v, v)

    pages = collect_pages()
    print(f"Collected {len(pages)} pages")

    def full_line(e):
        ptype = e.get("type", "?")
        archs = ", ".join(e.get("architectures", []))
        return f"- [{e.get('title', e.get('id', '?'))}]({link(e['path'])}) `[{ptype}]` arch:{archs}"

    def simple_line(e):
        return f"- [{e.get('title', e.get('id', '?'))}]({link(e['path'])})"

    def conf_line(e):
        conf = e.get("confidence", "?")
        archs = ", ".join(e.get("architectures", []))
        return f"- [{e.get('title', e.get('id', '?'))}]({link(e['path'])}) conf:{conf} arch:{archs}"

    # by-hardware: union of `tags` (normalized, restricted to hw vocab) and hardware_features
    hw_groups = defaultdict(list)
    for p in pages:
        for v in list(p.get("tags", [])) + list(p.get("hardware_features", [])):
            cv = norm(v)
            if cv in hw_vocab:
                hw_groups[cv].append(p)
    hw_lines = ["# Index: By Hardware Feature\n"]
    for key in sorted(hw_groups):
        entries = dedupe(hw_groups[key])
        hw_lines.append(f"\n## {key} ({len(entries)} pages)\n")
        for e in entries:
            hw_lines.append(full_line(e))

    indices = {
        "by-hardware-feature.md": "\n".join(hw_lines),
        "by-technique.md": grouped_index("Index: By Technique", pages, "techniques",
                                          normalize=norm, valid=tech_vocab, line_fn=simple_line),
        "by-kernel-type.md": grouped_index("Index: By Kernel Type", pages, "kernel_types",
                                           line_fn=conf_line),
        "by-language.md": grouped_index("Index: By Language", pages, "languages",
                                        line_fn=simple_line),
        "by-architecture.md": grouped_index("Index: By Architecture", pages, "architectures",
                                            valid=arch_vocab, line_fn=full_line),
        "by-problem.md": generate_by_problem(pages),
    }

    for name, content in indices.items():
        (QUERIES_DIR / name).write_text(content)
        print(f"  Generated: {name}")

    write_manifest(pages)
    print(f"\nDone: {len(indices)} indices + manifest generated in queries/")


def generate_by_problem(pages):
    lines = ["# Index: By Problem\n", "\nSymptom → Pattern → Technique → Solution\n"]
    patterns = [p for p in pages if p.get("type") == "wiki-pattern"]
    for p in sorted(patterns, key=lambda x: x.get("id", "")):
        symptoms = p.get("symptoms", p.get("tags", []))
        lines.append(f"\n### {p.get('title', '?')}\n")
        lines.append(f"- ID: `{p.get('id', '?')}`")
        lines.append(f"- Path: [{p['path']}]({link(p['path'])})")
        if symptoms:
            lines.append(f"- Tags: {', '.join(symptoms)}")
        related = p.get("related", [])
        if related:
            lines.append(f"- Related: {', '.join(f'`{r}`' for r in related)}")
    return "\n".join(lines)


def write_manifest(pages):
    """Compact machine-readable manifest for agents to load first."""
    records = []
    for p in sorted(pages, key=lambda x: x.get("id", x["path"])):
        records.append({
            "id": p.get("id"),
            "type": p.get("type"),
            "title": p.get("title"),
            "path": p["path"],
            "architectures": p.get("architectures", []),
            "tags": p.get("tags", []),
            "kernel_types": p.get("kernel_types", []),
            "languages": p.get("languages", []),
            "confidence": p.get("confidence"),
            "summary": (p.get("description") or "")[:160],
        })
    (QUERIES_DIR / "pages.json").write_text(json.dumps(records, indent=1))

    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"] or "?"].append(r)
    lines = ["# Page Manifest\n",
             f"\n{len(records)} pages. Machine-readable form: [pages.json](pages.json).\n"]
    for t in sorted(by_type):
        lines.append(f"\n## {t} ({len(by_type[t])})\n")
        for r in by_type[t]:
            lines.append(f"- `{r['id']}` — [{r['title']}]({link(r['path'])})")
    (QUERIES_DIR / "INDEX.md").write_text("\n".join(lines))
    print("  Generated: pages.json, INDEX.md")


if __name__ == "__main__":
    main()
