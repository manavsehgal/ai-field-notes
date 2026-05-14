# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""Artifact publishing primitives — HuggingFace Hub adapter + auto model card.

The Orionfold publishing surface: turn a `fieldkit.lineage` run plus a quant /
LoRA / adapter receipt into a HuggingFace repo with a deterministic model
card and a per-artifact manifest the source repo and destination site can both
read.

Three public surfaces:

- `ModelCard` — frontmatter + body builder. Renders the canonical card shape
  every Orionfold artifact gets: tags + license + base_model + library_name
  YAML, a body that includes Spark `tok/s` + perplexity + thermal envelope,
  a lineage block (if a `LineageStore` is provided), the `ollama pull` /
  `from_pretrained` one-liners, and a "Published by Orionfold LLC" footer
  with a backlink to ainative.business/field-notes/<slug>/.
- `ArtifactManifest` — frozen dataclass for the `src/content/artifacts/<slug>.yaml`
  Phase-2 sync record. Schema is intentionally flat (primitive types only) so
  the writer is hand-rolled stdlib YAML; matches the schema sketch in memory
  `project_artifact_manifests_phase2`. Mac destination renders catalog pages
  from `getCollection('artifacts')`.
- `HFHubAdapter` — thin wrapper around `huggingface_hub`. Lazy import (so
  `import fieldkit.publish` costs nothing without the dep). Defaults to
  `dry_run=True`: the adapter materializes the upload set to a staging
  directory and logs the would-be calls; flip `dry_run=False` to actually
  push. The dry-run path is fully testable offline.

Why this module exists. The MTBM pivot (per `ideas/mtbm-use-cases.md` §6
Pick #1) ships AI-native artifacts at a 3–5 day cadence. Manual model-card
authoring at that velocity is the bottleneck — every quant needs a tags list,
a perplexity table, a tok/s number, a thermal envelope note, a lineage
backlink. This module makes the card the deterministic output of the
quant+lineage run, not a hand-edit.

Companion modules:
- `fieldkit.lineage` (v0.3) — the trial log this module reads to render the
  "what's been tried" section of the card. Optional — cards work without one.
- `fieldkit.quant` (v0.4, sibling) — produces the `QuantReport` shape this
  module renders. Defined there to avoid a circular import; this module duck-types
  whatever has `.format`, `.variants`, `.perplexity`, `.tokens_per_sec`,
  `.sustained_load_minutes`, `.base_model`, `.lineage_run_id`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

__all__ = [
    "ArtifactKind",
    "ArtifactManifest",
    "HFHubAdapter",
    "HFHubNotAvailable",
    "HFAuthError",
    "ModelCard",
    "PublishError",
    "PublishResult",
    "publish_quant",
    "write_artifact_manifest",
    "ORIONFOLD_BRAND",
    "ORIONFOLD_HF_HANDLE",
    "ORIONFOLD_HF_ORG",
    "ARTIFACT_KINDS",
]


ORIONFOLD_BRAND: str = "Orionfold LLC"
"""Parent-company brand stamped on every model card footer.

Per memory `project_orionfold_parent_brand`: Orionfold LLC owns the AI-artifact
publishing surfaces (HF, Civitai, etc.); the ai-field-notes blog (published as
ainative.business/field-notes/) is an editorial property under it. The card
footer always says `Published by Orionfold LLC` with a link back to
`orionfold.com` and a cross-reference to the methods article.
"""

ORIONFOLD_HF_HANDLE: str = "Orionfold"
"""Default HuggingFace handle for Orionfold publishing surfaces.

Bartowski-shape: a single handle, repo names carry the format suffix
(`Orionfold/<model>-GGUF`, `Orionfold/<model>-LoRA`). The handle is the
existing personal user account `Orionfold` (the short org slug `orionfold`
was unavailable, and the fallback org `orionfoldllc` was abandoned on
2026-05-14 in favor of publishing under the user — Bartowski precedent).
The LLC parent brand survives off-HF; the HF surface is the user.
"""

# Back-compat alias — keep `ORIONFOLD_HF_ORG` callable as a deprecation shim
# in case any out-of-tree caller imports it. Drop on the next fieldkit cut.
ORIONFOLD_HF_ORG: str = ORIONFOLD_HF_HANDLE

ARTIFACT_KINDS: tuple[str, ...] = (
    "quant",
    "lora",
    "adapter",
    "embed",
    "reranker",
    "dataset",
    "space",
    "bench",
)
"""Manifest `kind` enum. Mirrors `project_artifact_manifests_phase2` schema."""


ArtifactKind = str
"""One of `ARTIFACT_KINDS`. Type-aliased as `str` since Python enums in the
public API would over-constrain the manifest reader on the destination side."""


# --- Errors --------------------------------------------------------------


class PublishError(Exception):
    """Base class for fieldkit.publish errors."""


class HFHubNotAvailable(PublishError, ImportError):
    """`huggingface_hub` is not installed in this env.

    The adapter lazy-imports the dep, so this only surfaces when you actually
    call into `HFHubAdapter` push paths. Install with `pip install
    huggingface_hub`. The dry-run path does not require it.
    """


class HFAuthError(PublishError):
    """Token missing or push refused. Check `HF_TOKEN` env or pass `token=` explicitly."""


# --- Model card --------------------------------------------------------------


@dataclass(frozen=True)
class ModelCard:
    """Frontmatter + body builder for a HuggingFace model card.

    Constructed by `publish_quant(...)` from a `QuantReport`-shaped object plus
    a base-model id. Renders to a single `README.md`-style string via `render()`.

    The card carries five sections in canonical order:

    1. YAML frontmatter — `language`, `license`, `library_name`, `base_model`,
       `tags`, `pipeline_tag`, `model_creator`. HuggingFace surfaces these on
       the model page directly.
    2. Title + one-line elevator.
    3. **Spark-tested** block — perplexity + tok/s + thermal envelope per
       variant. The differentiation from Bartowski/Unsloth/mradermacher (per
       Q1 + Q9 decisions 2026-05-12): every Orionfold quant card includes the
       Spark measurement triple.
    4. **Variants** table — the per-quant column (size, recommended use).
    5. **How to run** — `ollama pull` one-liner (GGUF only) and
       `from_pretrained(...)` snippet (HF Transformers path).
    6. **Lineage** — if a `LineageStore` was supplied, the rendered prompt block.
    7. **Methods** — link to `ainative.business/field-notes/<article_slug>/`.
    8. **Footer** — `Published by Orionfold LLC` + orionfold.com backlink.
    """

    title: str
    one_liner: str
    base_model: str
    license: str = "apache-2.0"
    library_name: str = "gguf"
    pipeline_tag: str = "text-generation"
    language: tuple[str, ...] = ("en",)
    tags: tuple[str, ...] = ()
    quant_format: str = "gguf"
    variants: tuple[dict[str, Any], ...] = ()
    perplexity: dict[str, float] = field(default_factory=dict)
    tokens_per_sec: dict[str, float] = field(default_factory=dict)
    sustained_load_minutes: Optional[float] = None
    vertical_eval: dict[str, float] = field(default_factory=dict)
    """Per-variant vertical-eval accuracy (FinanceBench / LegalBench / SemEval).
    Added in v0.4.x for the vertical-curator quant pattern — every Orionfold
    card carries a vertical-domain accuracy score, not just wikitext perplexity."""
    vertical_eval_name: Optional[str] = None
    """Display name for the vertical eval column (e.g.,
    "FinanceBench (n=50, numeric_match)")."""
    ollama_pull_handle: Optional[str] = None
    transformers_snippet: Optional[str] = None
    lineage_prompt: Optional[str] = None
    article_slug: Optional[str] = None
    article_title: Optional[str] = None
    model_creator: str = ORIONFOLD_BRAND
    extra_yaml: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Return the full README.md-style card as a single string."""
        return _render_model_card(self)


def _render_yaml_scalar(value: Any) -> str:
    """Hand-rolled YAML emitter for primitives. Module is stdlib-only by design."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    s = str(value)
    if not s:
        return '""'
    if re.search(r"[:\#\[\]\{\},&*!|>'\"%@`]", s) or s[0] in "?-" or s != s.strip():
        return json.dumps(s, ensure_ascii=False)
    return s


def _render_yaml_block(data: dict[str, Any], indent: int = 0) -> list[str]:
    """Emit a tiny YAML subset (flat keys, str / num / bool / list-of-str / dict-of-num)."""
    out: list[str] = []
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            if not value:
                out.append(f"{prefix}{key}: {{}}")
                continue
            out.append(f"{prefix}{key}:")
            out.extend(_render_yaml_block(value, indent + 1))
        elif isinstance(value, (list, tuple)):
            if not value:
                out.append(f"{prefix}{key}: []")
                continue
            out.append(f"{prefix}{key}:")
            for item in value:
                out.append(f"{prefix}  - {_render_yaml_scalar(item)}")
        else:
            out.append(f"{prefix}{key}: {_render_yaml_scalar(value)}")
    return out


def _render_model_card(card: ModelCard) -> str:
    fm: dict[str, Any] = {
        "license": card.license,
        "library_name": card.library_name,
        "base_model": card.base_model,
        "pipeline_tag": card.pipeline_tag,
        "model_creator": card.model_creator,
        "language": list(card.language),
        "tags": list(card.tags),
    }
    fm.update(card.extra_yaml)
    yaml_lines = ["---", *_render_yaml_block(fm), "---", ""]

    lines: list[str] = []
    lines.append(f"# {card.title}")
    lines.append("")
    lines.append(card.one_liner)
    lines.append("")

    # Spark-tested block — the Orionfold differentiator.
    if (
        card.tokens_per_sec
        or card.perplexity
        or card.sustained_load_minutes is not None
        or card.vertical_eval
    ):
        has_vertical = bool(card.vertical_eval)
        vertical_label = card.vertical_eval_name or "Vertical eval"
        lines.append("## Spark-tested")
        lines.append("")
        triple_or_quad = "measurement quad" if has_vertical else "measurement triple"
        axes = (
            "perplexity, sustained `tok/s`, thermal envelope, and"
            f" **{vertical_label}** accuracy"
            if has_vertical
            else "perplexity, sustained `tok/s`, and thermal envelope"
        )
        lines.append(
            f"Every Orionfold quant ships with a {triple_or_quad} on the NVIDIA"
            f" DGX Spark (GB10, 128 GB unified memory): {axes}. The numbers"
            f" below are the actual run, not a wishlist."
        )
        lines.append("")
        if card.tokens_per_sec or card.perplexity or has_vertical:
            if has_vertical:
                header = (
                    f"| Variant | Size | Perplexity (wikitext-2) | tok/s on Spark | {vertical_label} |"
                )
                sep = "|---|---|---|---|---|"
            else:
                header = "| Variant | Size | Perplexity (wikitext-2) | tok/s on Spark |"
                sep = "|---|---|---|---|"
            lines.append(header)
            lines.append(sep)
            seen: list[str] = []
            for variant in card.variants:
                name = variant.get("name", "")
                if not name:
                    continue
                seen.append(name)
                size = variant.get("size", "")
                ppl = card.perplexity.get(name)
                tps = card.tokens_per_sec.get(name)
                ppl_str = f"{ppl:.3f}" if isinstance(ppl, (int, float)) else "—"
                tps_str = f"{tps:.1f}" if isinstance(tps, (int, float)) else "—"
                if has_vertical:
                    ve = card.vertical_eval.get(name)
                    ve_str = f"{ve:.1%}" if isinstance(ve, (int, float)) else "—"
                    lines.append(f"| {name} | {size} | {ppl_str} | {tps_str} | {ve_str} |")
                else:
                    lines.append(f"| {name} | {size} | {ppl_str} | {tps_str} |")
            # Catch perplexity/tps entries with no matching variant row.
            extra_names = set(card.perplexity.keys()) | set(card.vertical_eval.keys())
            for name in extra_names:
                if name in seen:
                    continue
                ppl = card.perplexity.get(name)
                tps = card.tokens_per_sec.get(name)
                ppl_str = f"{ppl:.3f}" if isinstance(ppl, (int, float)) else "—"
                tps_str = f"{tps:.1f}" if isinstance(tps, (int, float)) else "—"
                if has_vertical:
                    ve = card.vertical_eval.get(name)
                    ve_str = f"{ve:.1%}" if isinstance(ve, (int, float)) else "—"
                    lines.append(f"| {name} | — | {ppl_str} | {tps_str} | {ve_str} |")
                else:
                    lines.append(f"| {name} | — | {ppl_str} | {tps_str} |")
                seen.append(name)
            lines.append("")
        if card.sustained_load_minutes is not None:
            lines.append(
                f"**Thermal envelope:** sustained-load minutes before thermal"
                f" throttle on a single GB10 = **{card.sustained_load_minutes:.0f}"
                f" min**. Beyond this, expect tok/s degradation; the duty-cycle"
                f" disclosure is per Orionfold's quant-card standard."
            )
            lines.append("")

    # Variants table (descriptive — recommended use per variant).
    if card.variants:
        lines.append("## Variants")
        lines.append("")
        lines.append("| Variant | Recommended use |")
        lines.append("|---|---|")
        for variant in card.variants:
            name = variant.get("name", "")
            rec = variant.get("recommended", "")
            if name:
                lines.append(f"| {name} | {rec} |")
        lines.append("")

    # How to run.
    lines.append("## How to run")
    lines.append("")
    if card.ollama_pull_handle:
        lines.append("Ollama (one-liner):")
        lines.append("")
        lines.append("```bash")
        lines.append(f"ollama pull {card.ollama_pull_handle}")
        lines.append("```")
        lines.append("")
    if card.transformers_snippet:
        lines.append("HuggingFace Transformers:")
        lines.append("")
        lines.append("```python")
        lines.append(card.transformers_snippet.rstrip())
        lines.append("```")
        lines.append("")

    if card.lineage_prompt:
        lines.append("## Lineage")
        lines.append("")
        lines.append(
            "The trial log behind this quant — what was tried, what was kept,"
            " what was discarded. Emitted from `fieldkit.lineage.LineageStore`"
            " on the build machine; included so future iterations can pick up"
            " mid-stream."
        )
        lines.append("")
        lines.append(card.lineage_prompt.rstrip())
        lines.append("")

    if card.article_slug:
        lines.append("## Methods")
        lines.append("")
        title = card.article_title or card.article_slug
        lines.append(
            f"Full methodology and Spark-side measurement protocol:"
            f" [{title}](https://ainative.business/field-notes/{card.article_slug}/)."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"Published by **{card.model_creator}** · [orionfold.com](https://orionfold.com)"
        f" · Methods documented at [ainative.business/field-notes](https://ainative.business/field-notes/)."
    )
    lines.append("")

    return "\n".join(yaml_lines) + "\n".join(lines) + "\n"


# --- Manifest --------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactManifest:
    """The `src/content/artifacts/<slug>.yaml` Phase-2 sync record.

    Source repo writes this on every push; Mac destination renders catalog
    pages from `getCollection('artifacts')`. Schema mirrors the sketch in
    memory `project_artifact_manifests_phase2`.

    Flat by design — only primitive types and dicts of primitives — so the
    YAML emitter stays hand-rolled stdlib and the Astro Zod schema on the
    destination side is trivial.
    """

    slug: str
    kind: ArtifactKind
    artifact_class: str  # "class" is reserved; renamed in YAML output via _to_yaml_dict
    base_model: str
    hf_repo: str
    variants: tuple[str, ...] = ()
    perplexity: dict[str, float] = field(default_factory=dict)
    spark_tokens_per_sec: dict[str, float] = field(default_factory=dict)
    sustained_load_minutes: Optional[float] = None
    vertical_eval: dict[str, float] = field(default_factory=dict)
    """Per-variant vertical-eval accuracy (FinanceBench / LegalBench / SemEval).
    Added in v0.4.x alongside `ModelCard.vertical_eval` for the vertical-curator
    pattern. Empty for non-vertical (cross-cutting) quants."""
    vertical_eval_name: Optional[str] = None
    """Display name for the vertical eval (e.g.,
    "FinanceBench (n=50, numeric_match)")."""
    lineage_run_id: Optional[str] = None
    license_tier: str = "free"
    license_commercial_tier: Optional[str] = None
    article: Optional[str] = None
    civitai_id: Optional[int] = None
    download_count: Optional[int] = None
    published_at: Optional[str] = None  # ISO 8601

    def to_yaml(self) -> str:
        """Render to the YAML shape the destination Astro collection expects."""
        return _render_artifact_manifest_yaml(self)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict round-trip — `artifact_class` is renamed to `class` in output."""
        d: dict[str, Any] = {
            "slug": self.slug,
            "kind": self.kind,
            "class": self.artifact_class,
            "base_model": self.base_model,
            "hf_repo": self.hf_repo,
            "variants": list(self.variants),
        }
        if self.perplexity:
            d["perplexity"] = dict(self.perplexity)
        if self.spark_tokens_per_sec:
            d["spark_tokens_per_sec"] = dict(self.spark_tokens_per_sec)
        if self.sustained_load_minutes is not None:
            d["sustained_load_minutes"] = self.sustained_load_minutes
        if self.vertical_eval:
            d["vertical_eval"] = dict(self.vertical_eval)
        if self.vertical_eval_name:
            d["vertical_eval_name"] = self.vertical_eval_name
        if self.lineage_run_id:
            d["lineage_run_id"] = self.lineage_run_id
        d["license"] = {"tier": self.license_tier}
        if self.license_commercial_tier:
            d["license"]["commercial_tier"] = self.license_commercial_tier
        if self.article:
            d["article"] = self.article
        if self.civitai_id is not None:
            d["civitai_id"] = self.civitai_id
        if self.download_count is not None:
            d["download_count"] = self.download_count
        if self.published_at:
            d["published_at"] = self.published_at
        return d


def _render_artifact_manifest_yaml(manifest: ArtifactManifest) -> str:
    lines = _render_yaml_block(manifest.to_dict())
    return "\n".join(lines) + "\n"


def write_artifact_manifest(
    manifest: ArtifactManifest,
    *,
    artifacts_dir: Union[str, Path],
) -> Path:
    """Write the manifest to `<artifacts_dir>/<slug>.yaml`.

    Creates the directory if missing. The path returned is the absolute path
    of the written file — callers can stage it for git commit alongside the
    article.
    """
    dest_dir = Path(artifacts_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{manifest.slug}.yaml"
    dest.write_text(manifest.to_yaml(), encoding="utf-8")
    return dest


# --- HF Hub adapter -------------------------------------------------------


def _require_huggingface_hub() -> Any:
    """Lazy `huggingface_hub` import with a clear error.

    Mirrors the pattern in `fieldkit.training._require_torch`.
    """
    try:
        import huggingface_hub  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HFHubNotAvailable(
            "fieldkit.publish HF push paths require huggingface_hub."
            " Install it: `pip install huggingface_hub>=0.24`."
            " The dry-run path does not require this import."
        ) from exc
    return huggingface_hub


@dataclass
class PublishResult:
    """Return shape of `publish_quant` and `HFHubAdapter.push_folder`."""

    hf_repo: str
    files_uploaded: tuple[str, ...]
    dry_run: bool
    manifest_path: Optional[Path] = None
    card_path: Optional[Path] = None
    hf_url: Optional[str] = None


class HFHubAdapter:
    """Thin wrapper around `huggingface_hub`. Defaults to dry-run.

    Two construction modes:

    - `HFHubAdapter(staging_dir=...)` — dry-run. Lays out the upload set on disk,
      logs the would-be calls. No HF imports required, no token required.
    - `HFHubAdapter(staging_dir=..., dry_run=False, token=...)` — live push.
      Lazy-imports `huggingface_hub`, requires a write-scoped token (or
      `HF_TOKEN` env). Each `push_folder(...)` translates to
      `huggingface_hub.HfApi().upload_folder(...)`.

    Token resolution order: explicit `token=` arg → `HF_TOKEN` env →
    `huggingface_hub`'s default (`~/.cache/huggingface/token`). If all three
    are absent and `dry_run=False`, raises `HFAuthError`.

    The staging directory is intentionally opaque — callers populate it
    however they like (the model card, variant files, anything else). The
    adapter uploads the entire directory tree as one HF commit.
    """

    def __init__(
        self,
        *,
        staging_dir: Union[str, Path],
        dry_run: bool = True,
        token: Optional[str] = None,
        org: str = ORIONFOLD_HF_HANDLE,
    ) -> None:
        self.staging_dir = Path(staging_dir)
        self.dry_run = dry_run
        self.token = token
        self.org = org
        self._logged_calls: list[dict[str, Any]] = []

    @property
    def logged_calls(self) -> tuple[dict[str, Any], ...]:
        """Read-only view of dry-run call log. Useful for tests + audit trail."""
        return tuple(self._logged_calls)

    def _resolve_token(self) -> Optional[str]:
        if self.token:
            return self.token
        import os
        env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if env:
            return env
        return None  # let huggingface_hub fall back to its cached login

    def repo_id(self, name: str) -> str:
        """Resolve `name` (already-fully-qualified or bare) to `<org>/<name>`."""
        if "/" in name:
            return name
        return f"{self.org}/{name}"

    def stage_file(self, src: Union[str, Path], rel: str) -> Path:
        """Copy `src` into `staging_dir/<rel>`. Returns the destination path."""
        import shutil
        src_path = Path(src)
        dest = self.staging_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        return dest

    def stage_text(self, text: str, rel: str) -> Path:
        """Write `text` into `staging_dir/<rel>`. Returns the destination path."""
        dest = self.staging_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return dest

    def push_folder(
        self,
        *,
        repo_name: str,
        commit_message: str = "Initial Orionfold upload",
        private: bool = False,
        repo_type: str = "model",
    ) -> PublishResult:
        """Push everything in `staging_dir` to HF as a single commit.

        Dry-run: enumerates the staged files, logs the call, returns a
        `PublishResult` with `dry_run=True`.

        Live: calls `huggingface_hub.HfApi().upload_folder(...)`.
        """
        repo_id = self.repo_id(repo_name)
        files = self._enumerate_staged()
        call = {
            "method": "upload_folder",
            "repo_id": repo_id,
            "folder_path": str(self.staging_dir),
            "commit_message": commit_message,
            "private": private,
            "repo_type": repo_type,
            "files": list(files),
        }
        self._logged_calls.append(call)

        if self.dry_run:
            return PublishResult(
                hf_repo=repo_id,
                files_uploaded=tuple(files),
                dry_run=True,
                hf_url=None,
            )

        hf = _require_huggingface_hub()
        token = self._resolve_token()
        api = hf.HfApi(token=token)
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                exist_ok=True,
            )
            commit_info = api.upload_folder(
                folder_path=str(self.staging_dir),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
        except Exception as exc:  # noqa: BLE001 - wrap any auth/network failure
            msg = str(exc).lower()
            if "401" in msg or "unauthorized" in msg or "token" in msg:
                raise HFAuthError(
                    f"HF push refused for {repo_id}: {exc}. Check HF_TOKEN env"
                    f" or pass token=... explicitly."
                ) from exc
            raise

        url = f"https://huggingface.co/{repo_id}"
        if hasattr(commit_info, "commit_url") and commit_info.commit_url:
            url = commit_info.commit_url
        return PublishResult(
            hf_repo=repo_id,
            files_uploaded=tuple(files),
            dry_run=False,
            hf_url=url,
        )

    def _enumerate_staged(self) -> list[str]:
        if not self.staging_dir.exists():
            return []
        return sorted(
            str(p.relative_to(self.staging_dir))
            for p in self.staging_dir.rglob("*")
            if p.is_file()
        )


# --- Orchestrator --------------------------------------------------------


def publish_quant(
    *,
    quant_report: Any,
    base_model: str,
    repo_name: str,
    staging_dir: Union[str, Path],
    artifacts_dir: Optional[Union[str, Path]] = None,
    article_slug: Optional[str] = None,
    article_title: Optional[str] = None,
    lineage_store: Any = None,
    lineage_run_id: Optional[str] = None,
    dry_run: bool = True,
    token: Optional[str] = None,
    org: str = ORIONFOLD_HF_HANDLE,
    extra_tags: Sequence[str] = (),
    ollama_pull_handle: Optional[str] = None,
    transformers_snippet: Optional[str] = None,
    vertical_eval: Optional[dict[str, float]] = None,
    vertical_eval_name: Optional[str] = None,
) -> PublishResult:
    """Orchestrate model-card render + manifest write + HF push.

    Designed to be the one-line caller from a Spark-side build script::

        from fieldkit.quant import quantize_gguf
        from fieldkit.publish import publish_quant

        report = quantize_gguf(model="instruction-pretrain/finance-Llama3-8B", outdir="/data/quants/...")
        publish_quant(
            quant_report=report,
            base_model="instruction-pretrain/finance-Llama3-8B",
            repo_name="finance-Llama3-8B-GGUF",
            staging_dir="/tmp/orionfold-stage/finance",
            artifacts_dir="/home/nvidia/ai-field-notes/src/content/artifacts",
            article_slug="becoming-a-gguf-publisher-on-spark",
            article_title="Vertical-curator quants on Spark — FinanceBench, day 7",
            vertical_eval={"Q4_K_M": 0.62, "Q5_K_M": 0.66, ...},
            vertical_eval_name="FinanceBench (n=50, numeric_match)",
            dry_run=True,
        )

    `quant_report` is duck-typed; any object with `format`, `variants`,
    `perplexity`, `tokens_per_sec`, `sustained_load_minutes`,
    `vertical_eval`, `vertical_eval_name`, `variant_files` works. (The
    canonical producer is `fieldkit.quant.QuantReport`; this module imports
    nothing from `fieldkit.quant` to avoid a circular dep.) Explicit kwargs
    `vertical_eval` / `vertical_eval_name` override whatever the report
    carries — handy when scoring happens out-of-band from quantization.
    """
    # Pull duck-typed fields off the report.
    quant_format = str(getattr(quant_report, "format", "gguf"))
    variants = tuple(getattr(quant_report, "variants", ()) or ())
    perplexity = dict(getattr(quant_report, "perplexity", {}) or {})
    tokens_per_sec = dict(getattr(quant_report, "tokens_per_sec", {}) or {})
    sustained = getattr(quant_report, "sustained_load_minutes", None)
    variant_files = dict(getattr(quant_report, "variant_files", {}) or {})
    if vertical_eval is None:
        vertical_eval = dict(getattr(quant_report, "vertical_eval", {}) or {})
    if vertical_eval_name is None:
        vertical_eval_name = getattr(quant_report, "vertical_eval_name", None)

    # Build the card.
    tag_set: list[str] = [
        quant_format,
        "spark-tested",
        "orionfold",
        f"base_model:{base_model}",
    ]
    tag_set.extend(extra_tags)
    library_name = "gguf" if quant_format == "gguf" else "transformers"

    title = f"{repo_name.replace('-', ' ').replace('/', ' / ')}"
    one_liner = (
        f"`{quant_format.upper()}` quantizations of `{base_model}`, "
        f"verified end-to-end on the NVIDIA DGX Spark (GB10, 128 GB unified memory)."
    )

    lineage_prompt: Optional[str] = None
    if lineage_store is not None:
        try:
            snap = lineage_store.render_prompt()
            lineage_prompt = getattr(snap, "rendered_markdown", None) or str(snap)
        except Exception:
            lineage_prompt = None

    variant_rows = tuple(
        {
            "name": v,
            "size": variant_files.get(v, {}).get("size") if isinstance(variant_files.get(v), dict) else "",
            "recommended": _default_variant_recommendation(v),
        }
        for v in variants
    )

    card = ModelCard(
        title=title,
        one_liner=one_liner,
        base_model=base_model,
        library_name=library_name,
        quant_format=quant_format,
        tags=tuple(tag_set),
        variants=variant_rows,
        perplexity=perplexity,
        tokens_per_sec=tokens_per_sec,
        sustained_load_minutes=sustained,
        vertical_eval=vertical_eval,
        vertical_eval_name=vertical_eval_name,
        ollama_pull_handle=ollama_pull_handle,
        transformers_snippet=transformers_snippet,
        lineage_prompt=lineage_prompt,
        article_slug=article_slug,
        article_title=article_title,
    )

    # Stage the card + variant files.
    adapter = HFHubAdapter(staging_dir=staging_dir, dry_run=dry_run, token=token, org=org)
    card_path = adapter.stage_text(card.render(), "README.md")
    for v, info in variant_files.items():
        if isinstance(info, dict) and "path" in info:
            src_path = Path(info["path"])
            if src_path.exists():
                rel = info.get("rel") or src_path.name
                adapter.stage_file(src_path, rel)

    # Manifest (Phase-2 sync).
    manifest_path: Optional[Path] = None
    if artifacts_dir is not None:
        slug = repo_name.lower().replace("/", "-")
        manifest = ArtifactManifest(
            slug=slug,
            kind="quant",
            artifact_class=quant_format,
            base_model=base_model,
            hf_repo=adapter.repo_id(repo_name),
            variants=variants,
            perplexity=perplexity,
            spark_tokens_per_sec=tokens_per_sec,
            sustained_load_minutes=sustained,
            vertical_eval=vertical_eval,
            vertical_eval_name=vertical_eval_name,
            lineage_run_id=lineage_run_id,
            article=f"articles/{article_slug}/" if article_slug else None,
            published_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        manifest_path = write_artifact_manifest(manifest, artifacts_dir=artifacts_dir)

    # Push (or dry-run).
    result = adapter.push_folder(repo_name=repo_name)
    result.card_path = card_path
    result.manifest_path = manifest_path
    return result


def _default_variant_recommendation(variant: str) -> str:
    """Map a GGUF variant name to a one-liner about when to use it."""
    table = {
        "Q4_K_M": "Best balance — fits comfortably in Spark unified memory at 70B; default pick.",
        "Q5_K_M": "Higher quality than Q4_K_M with modest size bump.",
        "Q6_K": "Near-lossless; recommended if memory headroom allows.",
        "Q8_0": "Effectively lossless; reach for this when quality matters more than throughput.",
        "F16": "Reference — no quantization. Use only for measurement / baseline.",
        "AWQ-int4": "AWQ activation-aware int4 for vLLM / TGI.",
        "GPTQ-int4": "GPTQ int4 calibrated weights.",
        "EXL3": "exllamav3 variable-bpw quant for high tok/s on consumer Ampere.",
        "MLX-4bit": "Apple Silicon (Metal) 4-bit.",
        "NVFP4": "NVIDIA FP4 — Blackwell/Hopper FP4-aware tensor cores.",
    }
    return table.get(variant, "Variant-specific use case TBD.")
