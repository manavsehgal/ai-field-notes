#!/usr/bin/env python3
# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""Build the patent-strategist raw corpus on disk.

Per `specs/patent-strategist-v1.md` §3.2 + §6: pulls commercial-safe patent
sources to `/home/nvidia/data/corpus/patent/<source>/*.jsonl` and writes a
provenance snapshot at `evidence/patent-strategist/corpus-snapshot.json`
that the R10 (license-drift) mitigation hangs on.

Sources (spec §6.1):

    | source       | license      | role                        | tier |
    | ------------ | ------------ | --------------------------- | ---- |
    | bigpatent    | CC-BY-4.0    | Family A drafting / style   | 1    |
    | patentmatch  | (see notes)  | Family B analytical primary | 1    |
    | mpep         | public dom.  | Family D anchor + RAG       | 2    |
    | oa           | public dom.  | Family D procedural primary | 3    |
    | gpat         | CC-BY-4.0    | Family C landscape          | 3    |

Tier 1 sources pull cleanly from HuggingFace (anonymous). Tier 2 (MPEP)
needs an HTTP scraper. Tier 3 (USPTO OARD + Google Patents BigQuery) needs
external auth (USPTO data portal + gcloud). Tier 1 runs by default; tier 2
runs when `--sources` includes ``mpep``; tier 3 sources are scaffolded but
write a stub snapshot entry with ``status: blocked`` so downstream tasks
can plan around the gap.

Spec drift note: the spec lists ``pakuvis/PatentMatch`` for the PatentMatch
pull, but that HF repo returns 404. ``BNNT/PatentMatch`` (Apache-2.0,
English+Chinese JSON) is the closest available substitute; we pull the
English half and flag the substitution in the snapshot. The canonical
HPI-Naumann PatentMatch lives at https://hpi.de/naumann/s/patentmatch but
distributes via direct download, not HF — wire that in once the dataset
shape is confirmed.

Usage::

    python scripts/build_patent_corpus.py                       # all default-enabled
    python scripts/build_patent_corpus.py --sources bigpatent   # one source
    python scripts/build_patent_corpus.py --max-per-source 2000 # cap rows
    python scripts/build_patent_corpus.py --force               # overwrite existing JSONLs

The script is idempotent: existing per-source output dirs are skipped
unless ``--force`` is passed. Snapshot is rewritten on every run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("/home/nvidia/data/corpus/patent")
SNAPSHOT_PATH = REPO_ROOT / "evidence" / "patent-strategist" / "corpus-snapshot.json"

# `~/.cache/huggingface/hub/` is root-owned on this Spark (artifact of an
# earlier container-run download). Redirect cache to user-writable space.
# Skip if caller has already set HF_HOME explicitly.
os.environ.setdefault("HF_HOME", "/home/nvidia/data/.hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "/home/nvidia/data/.hf-cache/hub")

ALL_SOURCES = ("bigpatent", "patentmatch", "mpep", "oa", "gpat")
DEFAULT_SOURCES = ("bigpatent", "patentmatch", "mpep")
TIER3_SOURCES = ("oa", "gpat")


@dataclass
class SourceResult:
    name: str
    status: str  # "pulled" | "pending" | "blocked" | "skipped"
    hf_repo: str | None = None
    commit_sha: str | None = None
    license: str | None = None
    rows: int = 0
    files: list[str] = field(default_factory=list)
    notes: str = ""


# --- BIGPATENT (Tier 1) -----------------------------------------------------


def pull_bigpatent(out_dir: Path, max_rows: int | None) -> SourceResult:
    """Pull BIGPATENT abstracts from HF `big_patent`.

    Pulls IPC configs `g` (physics) + `h` (electricity) — the two most
    relevant for the tech-leaning patent-strategist scope. Caps at
    `max_rows` per config to keep the working corpus manageable; spec mix
    only needs ~2,500 examples downstream so the cap can be aggressive.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    repo_id = "big_patent"
    sha = HfApi().dataset_info(repo_id).sha
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    files: list[str] = []
    for cfg in ("g", "h"):
        path = out_dir / f"bigpatent-{cfg}-train.jsonl"
        ds = load_dataset(repo_id, cfg, split="train", streaming=True)
        with path.open("w") as f:
            for i, row in enumerate(ds):
                if max_rows is not None and i >= max_rows:
                    break
                f.write(
                    json.dumps(
                        {
                            "patent_number": row.get("patent_number"),
                            "ipc_class": cfg,
                            "description": row.get("description", "")[:8000],
                            "abstract": row.get("abstract", ""),
                        }
                    )
                    + "\n"
                )
                total += 1
        files.append(path.name)
        print(f"  bigpatent[{cfg}] → {path.name}: {total} rows so far", flush=True)
    return SourceResult(
        name="bigpatent",
        status="pulled",
        hf_repo=repo_id,
        commit_sha=sha,
        license="CC-BY-4.0",
        rows=total,
        files=files,
        notes="configs g+h (physics+electricity); abstracts capped to 8k chars",
    )


# --- PatentMatch substitute (Tier 1) ---------------------------------------


def pull_patentmatch(out_dir: Path, max_rows: int | None) -> SourceResult:
    """Pull `BNNT/PatentMatch` (Apache-2.0) as a substitute for the spec's
    `pakuvis/PatentMatch` (404 on HF as of 2026-05-17).

    The English JSON is loaded directly; the canonical HPI-Naumann
    PatentMatch (6.2M EPO pairs) ships outside HF and needs follow-up.
    """
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]

    repo_id = "BNNT/PatentMatch"
    sha = HfApi().dataset_info(repo_id).sha
    out_dir.mkdir(parents=True, exist_ok=True)

    src = hf_hub_download(repo_id, "PatentMatch_en.json", repo_type="dataset")
    # Despite the `.json` extension the file is newline-delimited JSON
    # (one record per line — instruction-tuning shape).
    rows: list[dict] = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break

    path = out_dir / "patentmatch-en.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return SourceResult(
        name="patentmatch",
        status="pulled",
        hf_repo=repo_id,
        commit_sha=sha,
        license="Apache-2.0",
        rows=len(rows),
        files=[path.name],
        notes=(
            "SUBSTITUTE: spec's pakuvis/PatentMatch returns 404; BNNT/PatentMatch "
            "used instead (500 rows, instruction-tuning shape — small for Family B "
            "primary). Canonical HPI-Naumann PatentMatch (6.2M EPO pairs) needs "
            "direct-download wiring before W2 bench preflight."
        ),
    )


# --- MPEP (Tier 2) ----------------------------------------------------------


MPEP_INDEX_URL = "https://mpep.uspto.gov/RDMS/MPEP/current"


def pull_mpep(out_dir: Path, max_rows: int | None) -> SourceResult:
    """Scrape USPTO MPEP section text from the eMPEP HTML index.

    The eMPEP system serves ~2K sections under
    https://mpep.uspto.gov/RDMS/MPEP/current. Scraping is gated behind a
    JS-rendered chrome layer in 2026, so this function currently stubs
    out — wiring requires either (a) a Playwright headless render of the
    section TOC then per-section HTML fetch, or (b) a one-shot USPTO bulk
    download (PDF/XML) parsed offline. Both paths are W1 follow-up work.
    """
    return SourceResult(
        name="mpep",
        status="pending",
        license="public-domain (17 USC §105)",
        notes=(
            "Scraper stub. eMPEP TOC is JS-rendered; need Playwright pass or "
            "USPTO bulk download. See MPEP_INDEX_URL constant."
        ),
    )


# --- USPTO OARD (Tier 3) ----------------------------------------------------


def pull_oa(out_dir: Path, max_rows: int | None) -> SourceResult:
    return SourceResult(
        name="oa",
        status="blocked",
        license="public-domain (17 USC §105)",
        notes=(
            "USPTO Office Action Research Dataset (4.4M actions 2008-2017). "
            "Download via USPTO data portal (https://www.uspto.gov/ip-policy/"
            "economic-research/research-datasets/) — needs portal account + "
            "bulk file fetch (CSV/Stata, multi-GB). Wire in W1-late."
        ),
    )


# --- Google Patents BigQuery (Tier 3) --------------------------------------


def pull_gpat(out_dir: Path, max_rows: int | None) -> SourceResult:
    return SourceResult(
        name="gpat",
        status="blocked",
        license="CC-BY-4.0",
        notes=(
            "Google Patents Public Data BigQuery dataset (patents-public-data). "
            "Needs gcloud auth + google-cloud-bigquery + a US-only IPC-class "
            "filtered query (Family C landscape work). Wire in W2 alongside "
            "the RAG index build."
        ),
    )


# --- Dispatch ---------------------------------------------------------------


PULLERS: dict[str, Callable[[Path, int | None], SourceResult]] = {
    "bigpatent": pull_bigpatent,
    "patentmatch": pull_patentmatch,
    "mpep": pull_mpep,
    "oa": pull_oa,
    "gpat": pull_gpat,
}


def _iter_existing(path: Path) -> Iterator[Path]:
    if path.exists():
        yield from path.glob("*.jsonl")


def _build_snapshot(results: list[SourceResult]) -> dict[str, Any]:
    return {
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_ref": "specs/patent-strategist-v1.md §6",
        "sources": {
            r.name: {
                "status": r.status,
                "hf_repo": r.hf_repo,
                "commit_sha": r.commit_sha,
                "license": r.license,
                "rows": r.rows,
                "files": r.files,
                "notes": r.notes,
            }
            for r in results
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated source names. Available: {','.join(ALL_SOURCES)}. "
        f"Default: {','.join(DEFAULT_SOURCES)}.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root. Default: {DEFAULT_OUT}",
    )
    p.add_argument(
        "--max-per-source",
        type=int,
        default=5000,
        help="Row cap per source/config. Default: 5000.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite per-source dirs that already have JSONLs.",
    )
    args = p.parse_args()

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in requested if s not in PULLERS]
    if unknown:
        print(f"FATAL: unknown sources: {unknown}", file=sys.stderr)
        return 2

    results: list[SourceResult] = []
    for src in requested:
        src_dir = args.out_dir / src
        existing = list(_iter_existing(src_dir))
        if existing and not args.force:
            print(f"[{src}] SKIP — {len(existing)} JSONLs already present (use --force to overwrite)")
            results.append(
                SourceResult(
                    name=src,
                    status="skipped",
                    rows=sum(1 for f in existing for _ in f.open()),
                    files=[f.name for f in existing],
                    notes="existing artifacts kept; pass --force to rebuild",
                )
            )
            continue
        print(f"[{src}] PULL → {src_dir}/", flush=True)
        try:
            results.append(PULLERS[src](src_dir, args.max_per_source))
        except Exception as exc:  # noqa: BLE001
            print(f"[{src}] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append(
                SourceResult(
                    name=src,
                    status="blocked",
                    notes=f"pull failed: {type(exc).__name__}: {exc}",
                )
            )

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snap = _build_snapshot(results)
    SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2) + "\n")
    print(f"\nSNAPSHOT → {SNAPSHOT_PATH}")
    for r in results:
        print(f"  {r.name:12s} {r.status:8s} rows={r.rows:>7d} {r.notes[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
