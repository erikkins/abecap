#!/usr/bin/env python3
"""
Refresh performance citations across the codebase.

Reads canonical numbers from `docs/canonical_numbers.json` and a surface map
from `scripts/perf_citations_surface_map.json`, then patches every regex-based
surface and emits a manual checklist for surfaces too complex for safe
auto-replacement (table restructures, FAQ rewrites, HTML doc edits).

Usage:
    # Dry run (default) — show diffs, no writes
    python3 scripts/refresh_perf_citations.py

    # Apply changes to the regex-patchable surfaces
    python3 scripts/refresh_perf_citations.py --apply

    # Use a specific canonical numbers file
    python3 scripts/refresh_perf_citations.py --canonical docs/canonical_numbers.json

After running with --apply:
- Review the unified diffs in the script's output.
- Walk the manual checklist written to /tmp/refresh_manual_checklist.md.
- Walk the out-of-repo checklist (also in the manual file).
- Commit with: "Refresh performance citations to <vintage>"
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DEFAULT = REPO_ROOT / "docs" / "canonical_numbers.json"
SURFACE_MAP_DEFAULT = REPO_ROOT / "scripts" / "perf_citations_surface_map.json"
MANUAL_CHECKLIST_OUT = Path("/tmp/refresh_manual_checklist.md")


class Color:
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def render_template(template: str, canonical: dict) -> str:
    """Apply Python format-string substitution using canonical_numbers.json keys."""
    try:
        return template.format(**canonical)
    except KeyError as e:
        return f"<<MISSING_KEY: {e}>>"


def show_diff(path: Path, before: str, after: str) -> None:
    if before == after:
        return
    diff = difflib.unified_diff(
        before.splitlines(keepends=False),
        after.splitlines(keepends=False),
        fromfile=str(path) + " (before)",
        tofile=str(path) + " (after)",
        n=2,
        lineterm="",
    )
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            print(f"{Color.BOLD}{line}{Color.RESET}")
        elif line.startswith("+"):
            print(f"{Color.GREEN}{line}{Color.RESET}")
        elif line.startswith("-"):
            print(f"{Color.RED}{line}{Color.RESET}")
        elif line.startswith("@"):
            print(f"{Color.BLUE}{line}{Color.RESET}")
        else:
            print(line)


def process_block_entry(entry: dict, canonical: dict, apply: bool) -> dict:
    """Block replacement between start_marker and end_marker (literal strings).
    Replaces everything between (exclusive of markers) with rendered template.
    Idempotent: running twice produces the same result.
    """
    rel_file = entry["file"]
    abs_file = REPO_ROOT / rel_file
    if not abs_file.exists():
        return {"id": entry["id"], "status": "MISSING_FILE", "file": rel_file}

    text = abs_file.read_text(encoding="utf-8")
    start = entry["start_marker"]
    end = entry["end_marker"]

    s_idx = text.find(start)
    e_idx = text.find(end, s_idx + len(start)) if s_idx >= 0 else -1
    if s_idx < 0 or e_idx < 0:
        return {
            "id": entry["id"],
            "status": "NO_MATCH",
            "file": rel_file,
            "find": f"start='{start}' end='{end}'",
        }

    rendered = render_template(entry["replace_template"], canonical)
    new_text = text[: s_idx + len(start)] + rendered + text[e_idx:]
    if new_text == text:
        return {"id": entry["id"], "status": "NOOP", "file": rel_file}

    if apply:
        abs_file.write_text(new_text, encoding="utf-8")

    return {
        "id": entry["id"],
        "status": "APPLIED" if apply else "DRY_RUN_CHANGED",
        "file": rel_file,
        "before": text,
        "after": new_text,
    }


def process_regex_entry(entry: dict, canonical: dict, apply: bool) -> dict:
    """Returns a result dict with status/details. Mutates file if apply=True."""
    rel_file = entry["file"]
    abs_file = REPO_ROOT / rel_file
    if not abs_file.exists():
        return {"id": entry["id"], "status": "MISSING_FILE", "file": rel_file}

    text = abs_file.read_text(encoding="utf-8")
    pattern = re.compile(entry["find"])
    replacement = render_template(entry["replace"], canonical)

    matches = list(pattern.finditer(text))
    if len(matches) == 0:
        return {
            "id": entry["id"],
            "status": "NO_MATCH",
            "file": rel_file,
            "find": entry["find"],
        }
    allow_multi = entry.get("allow_multi", False)
    if len(matches) > 1 and not allow_multi:
        return {
            "id": entry["id"],
            "status": "MULTI_MATCH",
            "file": rel_file,
            "find": entry["find"],
            "match_count": len(matches),
        }

    sub_count = 0 if allow_multi else 1
    new_text = pattern.sub(replacement, text, count=sub_count)
    if new_text == text:
        return {"id": entry["id"], "status": "NOOP", "file": rel_file}

    if apply:
        abs_file.write_text(new_text, encoding="utf-8")

    return {
        "id": entry["id"],
        "status": "APPLIED" if apply else "DRY_RUN_CHANGED",
        "file": rel_file,
        "before": text,
        "after": new_text,
    }


def render_manual_checklist(manual_entries: list, out_of_repo: list, canonical: dict, regex_misses: list) -> str:
    lines = []
    lines.append(f"# Performance Citation Refresh — Manual Checklist")
    lines.append("")
    lines.append(f"**Vintage:** {canonical.get('vintage')} ({canonical.get('vintage_label')})")
    lines.append(f"**Source:** {canonical.get('vintage_source')}")
    lines.append("")
    lines.append("Generated by `scripts/refresh_perf_citations.py`. The auto-patcher handled the regex surfaces; the items below need a human edit.")
    lines.append("")

    if regex_misses:
        lines.append("## ⚠ Auto-patch misses — investigate before commit")
        lines.append("")
        lines.append("These regex entries didn't match (file likely changed structure since the surface map was written). Review and update the surface map regex, or do a manual edit.")
        lines.append("")
        for m in regex_misses:
            lines.append(f"- **{m['id']}** in `{m['file']}` — status: `{m['status']}`")
            if m.get("find"):
                lines.append(f"  - find regex: `{m['find']}`")
            if m.get("match_count"):
                lines.append(f"  - matched {m['match_count']} times (regex not specific enough)")
        lines.append("")

    lines.append("## In-repo manual edits required")
    lines.append("")
    if not manual_entries:
        lines.append("_(none)_")
    for entry in manual_entries:
        lines.append(f"### {entry['id']}")
        lines.append(f"**File:** `{entry['file']}`  ")
        lines.append(f"**Lines:** {entry.get('line_hint', '?')}  ")
        lines.append(f"**Description:** {entry['description']}")
        lines.append("")
        if entry.get("current"):
            lines.append(f"**Current:** {entry['current']}")
            lines.append("")
        target = entry.get("target", "")
        target_rendered = render_template(target, canonical) if "{" in target else target
        if target_rendered:
            lines.append(f"**Target:** {target_rendered}")
            lines.append("")
        if entry.get("guidance"):
            lines.append(f"**Guidance:** {entry['guidance']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Out-of-repo surfaces")
    lines.append("")
    lines.append("These can't be auto-patched. Walk the list manually after the in-repo edits land.")
    lines.append("")
    for ext in out_of_repo:
        lines.append(f"- [ ] **{ext['surface']}**")
        lines.append(f"  - {ext['what_to_check']}")
        if ext.get("url"):
            lines.append(f"  - Location: `{ext['url']}`")
    lines.append("")

    lines.append("## Canonical numbers reference (this run)")
    lines.append("")
    headline_keys = [
        "wf_5y_avg_total_return_pct_rounded",
        "wf_5y_avg_annualized_pct",
        "wf_5y_avg_annualized_pct_marketing",
        "wf_5y_avg_sharpe",
        "wf_5y_avg_max_drawdown_pct",
        "wf_5y_worst_start_pct_rounded",
        "wf_5y_best_start_pct_rounded",
        "wf_5y_spy_avg_pct_rounded",
    ]
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for k in headline_keys:
        v = canonical.get(k)
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canonical", default=str(CANONICAL_DEFAULT), help="Path to canonical_numbers.json")
    parser.add_argument("--surface-map", default=str(SURFACE_MAP_DEFAULT), help="Path to surface_map.json")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file diffs (summary only)")
    args = parser.parse_args()

    canonical_path = Path(args.canonical)
    surface_map_path = Path(args.surface_map)

    if not canonical_path.exists():
        print(f"{Color.RED}canonical_numbers.json not found at {canonical_path}{Color.RESET}", file=sys.stderr)
        sys.exit(1)
    if not surface_map_path.exists():
        print(f"{Color.RED}surface_map.json not found at {surface_map_path}{Color.RESET}", file=sys.stderr)
        sys.exit(1)

    canonical = json.loads(canonical_path.read_text())
    surface_map = json.loads(surface_map_path.read_text())

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{Color.BOLD}{mode}{Color.RESET} — vintage={canonical.get('vintage')} ({canonical.get('vintage_label')})")
    print()

    regex_entries = [e for e in surface_map["entries"] if e.get("type") == "regex"]
    block_entries = [e for e in surface_map["entries"] if e.get("type") == "block"]
    manual_entries = [e for e in surface_map["entries"] if e.get("type") == "manual"]

    print(f"{len(regex_entries)} regex surfaces, {len(block_entries)} block surfaces, {len(manual_entries)} manual surfaces.")
    print()

    results = []
    for entry in regex_entries:
        result = process_regex_entry(entry, canonical, apply=args.apply)
        results.append(result)
        status = result["status"]
        sym = {
            "APPLIED":            f"{Color.GREEN}✓ APPLIED       {Color.RESET}",
            "DRY_RUN_CHANGED":    f"{Color.GREEN}✓ would change  {Color.RESET}",
            "NO_MATCH":           f"{Color.YELLOW}✗ NO MATCH      {Color.RESET}",
            "MULTI_MATCH":        f"{Color.YELLOW}✗ MULTI MATCH   {Color.RESET}",
            "NOOP":               f"{Color.DIM}· no-op        {Color.RESET}",
            "MISSING_FILE":       f"{Color.RED}✗ MISSING FILE {Color.RESET}",
        }.get(status, f"? {status}")
        print(f"  {sym} {result['id']:<45s} {result['file']}")

        if not args.quiet and "before" in result and "after" in result:
            show_diff(REPO_ROOT / result["file"], result["before"], result["after"])
            print()

    for entry in block_entries:
        result = process_block_entry(entry, canonical, apply=args.apply)
        results.append(result)
        status = result["status"]
        sym = {
            "APPLIED":            f"{Color.GREEN}✓ APPLIED       {Color.RESET}",
            "DRY_RUN_CHANGED":    f"{Color.GREEN}✓ would change  {Color.RESET}",
            "NO_MATCH":           f"{Color.YELLOW}✗ NO MATCH      {Color.RESET}",
            "NOOP":               f"{Color.DIM}· no-op        {Color.RESET}",
            "MISSING_FILE":       f"{Color.RED}✗ MISSING FILE {Color.RESET}",
        }.get(status, f"? {status}")
        print(f"  {sym} {result['id']:<45s} {result['file']} (block)")

        if not args.quiet and "before" in result and "after" in result:
            show_diff(REPO_ROOT / result["file"], result["before"], result["after"])
            print()

    misses = [r for r in results if r["status"] in ("NO_MATCH", "MULTI_MATCH", "MISSING_FILE")]
    applied = [r for r in results if r["status"] in ("APPLIED", "DRY_RUN_CHANGED")]

    print()
    print(f"{Color.BOLD}Summary:{Color.RESET}")
    print(f"  {Color.GREEN}{len(applied)} regex changes{' applied' if args.apply else ' would apply'}{Color.RESET}")
    print(f"  {Color.YELLOW}{len(misses)} regex misses{Color.RESET} (review before commit)")
    print(f"  {Color.BLUE}{len(manual_entries)} manual surfaces{Color.RESET} (see checklist)")

    checklist = render_manual_checklist(manual_entries, surface_map.get("out_of_repo_checklist", []), canonical, misses)
    MANUAL_CHECKLIST_OUT.write_text(checklist)
    print()
    print(f"  Manual checklist: {Color.BOLD}{MANUAL_CHECKLIST_OUT}{Color.RESET}")

    if args.apply and misses:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
