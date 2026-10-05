#!/usr/bin/env python3
"""
Validate all pages in ROCm-KernelWiki-Q against the schema and controlled vocabulary.

Checks (ERROR = fails the build, WARN = should be cleaned up):
  ERROR  missing required field, unknown page type, invalid architecture,
         invalid confidence, id prefix not matching type, duplicate id,
         id not matching filename for PR pages, incomplete performance_claim
  WARN   off-vocabulary tag/kernel_type/language/technique/hardware_feature,
         alias used instead of canonical term, malformed date, non-http url,
         source-pr with empty kernel_types AND languages, source-pr stuck on the
         default [cdna2,cdna3,cdna4] arch triple, unresolved sources/related ref,
         unresolved performance_claims source_id

Usage:
    python3 scripts/validate.py            # fail on ERRORs or warning-budget regressions
    python3 scripts/validate.py --strict   # exit non-zero on ERRORs or WARNs
    python3 scripts/validate.py --strict path/to/page.md [...]
    python3 scripts/validate.py --quiet    # only print the summary
"""

import argparse
import re
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WIKI_DIR = ROOT / "wiki"
SOURCES_DIR = ROOT / "sources"

DEFAULT_ARCH_TRIPLE = ["cdna2", "cdna3", "cdna4"]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def extract_frontmatter(filepath):
    with open(filepath) as f:
        content = f.read()
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None, content
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None, content
    return fm, content


def build_vocab(tags):
    """Return (canonical_structured, full_tag_vocab, aliases) sets/dicts."""
    archs = {a["id"] for a in tags.get("architectures", [])}
    hw = set(tags.get("hardware_features", {}).keys())
    tech = set(tags.get("techniques", {}).keys())
    kt = set(tags.get("kernel_types", []))
    langs = set(tags.get("languages", []))
    aux = set(tags.get("aux_tags", []))
    aliases = tags.get("aliases", {}) or {}

    structured = archs | hw | tech | kt | langs
    tag_vocab = structured | aux | set(aliases.keys())
    return {
        "archs": archs, "hw": hw, "tech": tech, "kt": kt, "langs": langs,
        "aux": aux, "aliases": aliases, "structured": structured,
        "tag_vocab": tag_vocab,
    }


def collect_all_ids():
    """Map id -> [paths]; a list reveals duplicates."""
    ids = {}
    for dir_path in [WIKI_DIR, SOURCES_DIR]:
        if not dir_path.exists():
            continue
        for md_file in dir_path.rglob("*.md"):
            fm, _ = extract_frontmatter(md_file)
            if fm and "id" in fm:
                ids.setdefault(fm["id"], []).append(md_file.relative_to(ROOT))
    return ids


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def validate_page(filepath, schemas, vocab, all_ids):
    errors, warnings = [], []

    def err(msg):
        errors.append(msg)

    def warn(msg):
        warnings.append(msg)

    fm, content = extract_frontmatter(filepath)
    if fm is None:
        return ["No YAML frontmatter found"], []

    page_type = fm.get("type", "")
    page_type_def = schemas.get("page_types", {}).get(page_type)
    if not page_type_def:
        return [f"Unknown page type: {page_type!r}"], []

    # Required fields
    for field in page_type_def.get("required_fields", []):
        if field not in fm:
            err(f"missing required field: {field}")

    # id <-> type prefix
    expected_prefix = page_type_def.get("id_prefix", "")
    page_id = fm.get("id", "")
    if expected_prefix and page_id and not page_id.startswith(expected_prefix):
        err(f"id {page_id!r} does not start with {expected_prefix!r} for type {page_type}")

    # duplicate id
    if page_id and len(all_ids.get(page_id, [])) > 1:
        others = ", ".join(str(p) for p in all_ids[page_id])
        err(f"duplicate id {page_id!r} also used by: {others}")

    # architectures (ERROR on invalid value)
    for arch in _as_list(fm.get("architectures")):
        if arch not in vocab["archs"]:
            err(f"invalid architecture: {arch} (valid: {sorted(vocab['archs'])})")

    # controlled-vocab list fields (WARN on off-vocab / alias)
    field_vocab = {
        "tags": vocab["tag_vocab"],
        "kernel_types": vocab["kt"],
        "languages": vocab["langs"],
        "techniques": vocab["tech"],
        "hardware_features": vocab["hw"],
    }
    for field, allowed in field_vocab.items():
        for val in _as_list(fm.get(field)):
            if val in vocab["aliases"]:
                warn(f"{field}: {val!r} is an alias; use {vocab['aliases'][val]!r}")
            elif val not in allowed:
                warn(f"{field}: off-vocabulary value {val!r}")

    # confidence (ERROR on invalid)
    valid_conf = {"verified", "source-reported", "inferred", "experimental"}
    conf = fm.get("confidence", "")
    if conf and conf not in valid_conf:
        err(f"invalid confidence: {conf}")

    # date format (WARN)
    date_val = fm.get("date")
    if date_val and not DATE_RE.match(str(date_val)):
        warn(f"date not YYYY-MM-DD: {date_val!r}")

    # url sanity (WARN)
    url = fm.get("url")
    if url and not str(url).startswith("http"):
        warn(f"url does not look like a link: {url!r}")

    # link integrity (WARN)
    for field in ["sources", "related", "landed_via"]:
        for ref in _as_list(fm.get(field)):
            if ref not in all_ids:
                warn(f"{field}: unresolved reference {ref!r}")

    # source-pr specific quality gates (WARN)
    if page_type == "source-pr":
        if not _as_list(fm.get("kernel_types")) and not _as_list(fm.get("languages")):
            warn("source-pr has empty kernel_types AND languages (un-retrievable by those indices)")
        if sorted(_as_list(fm.get("architectures"))) == DEFAULT_ARCH_TRIPLE:
            warn("source-pr still on default [cdna2,cdna3,cdna4] arch triple")
        if str(fm.get("captured_at", "")) >= "2026-07-29":
            if not fm.get("base_branch"):
                warn("new source-pr is missing base_branch")
            if fm.get("status") == "merged" and not fm.get("merge_commit"):
                warn("new merged source-pr is missing merge_commit")
        merge_commit = str(fm.get("merge_commit", ""))
        if merge_commit and not re.fullmatch(r"[0-9a-f]{40}", merge_commit):
            warn("merge_commit is not a 40-character lowercase hex SHA")

    # performance_claims structure (ERROR) + source_id resolution (WARN)
    if page_type == "wiki-kernel":
        for i, claim in enumerate(fm.get("performance_claims", []) or []):
            for field in ["gpu", "dtype", "metric", "value", "source_id"]:
                if field not in claim:
                    err(f"performance_claim[{i}] missing: {field}")
            sid = claim.get("source_id")
            if sid and sid not in all_ids:
                warn(f"performance_claim[{i}] source_id unresolved: {sid!r}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate ROCm-KernelWiki-Q pages")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    parser.add_argument("paths", nargs="*", help="optional Wiki/source pages to validate")
    args = parser.parse_args()

    schemas = load_yaml(DATA_DIR / "schemas.yaml")
    vocab = build_vocab(load_yaml(DATA_DIR / "tags.yaml"))
    quality_baseline = load_yaml(DATA_DIR / "quality-baseline.yaml")
    max_warnings = quality_baseline.get("validation", {}).get("max_warnings")
    all_ids = collect_all_ids()

    total_errors = total_warnings = total_pages = 0

    if args.paths:
        pages = []
        for value in args.paths:
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            try:
                path.relative_to(ROOT)
            except ValueError:
                parser.error(f"path is outside the repository: {value}")
            if not path.is_file() or path.suffix != ".md":
                parser.error(f"not a Markdown page: {value}")
            pages.append(path)
    else:
        pages = [
            md_file
            for dir_path in [WIKI_DIR, SOURCES_DIR]
            if dir_path.exists()
            for md_file in sorted(dir_path.rglob("*.md"))
        ]

    for md_file in pages:
        total_pages += 1
        errors, warnings = validate_page(md_file, schemas, vocab, all_ids)
        rel = md_file.relative_to(ROOT)
        if not args.quiet:
            for e in errors:
                print(f"  ERROR  {rel}: {e}")
            for w in warnings:
                print(f"  WARN   {rel}: {w}")
        total_errors += len(errors)
        total_warnings += len(warnings)

    print(f"\n{'='*60}")
    print(f"Validated: {total_pages} pages")
    print(f"Errors:    {total_errors}")
    print(f"Warnings:  {total_warnings}")
    if max_warnings is not None:
        print(f"Budget:    {max_warnings} warnings")
    print(f"Known IDs: {len(all_ids)}")
    print(f"{'='*60}")

    if total_errors > 0:
        return 1
    if max_warnings is not None and total_warnings > int(max_warnings):
        print(
            f"Warning budget exceeded: {total_warnings} > {max_warnings}",
            file=sys.stderr,
        )
        return 1
    if args.strict and total_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
