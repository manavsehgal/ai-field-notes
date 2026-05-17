# Card polish — the Orionfold engagement-pull recipe

This file codifies the **v5 §3.15.b engagement-pull recipe** — the set of model-card design moves that separates a card someone clones-and-forgets from a card someone *likes, follows, and recommends*. The reference exists because empirical evidence (Pulse #1, 2026-05-16) showed the four shipped Orionfold cards landed **472 DL / 0 likes** despite measurement quality being uniformly strong: the gap is a card-design problem, not a content problem.

The renderer in `fieldkit.publish.publish_quant` controls the bones of the card; this file owns the **above-the-fold differentiator block, the cross-link template, the wire-back hooks, and the metadata-completeness contract** that the renderer can't infer from kwargs alone. Read this every time you're about to push a new Orionfold artifact OR auditing an existing card via `card-audit` mode.

## The five engagement-pull elements

Every Orionfold card lands with all five present. Missing any one is the signal that triggers a card-polish loop before push (or a `card-audit` retro-fix for already-pushed cards).

### 1. Spark-tested differentiator block — at the top, not buried

The `## Spark-tested` section is the Orionfold moat. **Position it above `## How to run`, immediately after the one-liner.** That's the order the renderer emits today; preserve it. If anyone reshapes the card to put "How to run" first ("for usability"), reject — `## Spark-tested` is what differentiates Orionfold from the 50 other GGUF re-uploads of the same base. A user who scrolls past it hasn't seen the value prop.

Inside the block, the contract is:
- One short paragraph that names the measurement quad/triple (perplexity, sustained tok/s, thermal envelope, +/- vertical-eval) and says "the actual run, not a wishlist"
- The measurement table — Variant | Size | Perplexity | tok/s | (optional vertical-eval) — populated with `as-measured` numbers per `[[project_q8_anomaly_model_specific]]` (never pre-correct Q8_0)
- A `**Recommended:** <variant>` line ABOVE or IN the table (not buried under), keyed off `recommended_variant` (default `Q5_K_M`)

`scripts/verify_stage.sh` Check 3 enforces the table shape; this reference enforces the placement + recommended-variant prominence.

### 2. Sibling Orionfold card cross-links — explicit, not buried

Every card includes a `## Other Orionfold vertical curators` block at the end (above `## License`). The block lists each sibling card with a one-line "what it is, who it's for" hook. This is the single largest amplification lever:
- A user who landed on `Orionfold/II-Medical-8B-GGUF` for the medical card sees, in 30 seconds, that there are 3 other verticals — finance, legal, cyber — and that they're built on the same Spark-tested differentiator
- Each visit to one card threads visits to the others; download counts compound

Template (the publish_quant caller should pass via `extra_yaml` / a custom markdown block; the renderer doesn't auto-generate this yet — Phase 2 of `fieldkit.publish` may codify it):

```markdown
## Other Orionfold vertical curators

Same Spark-tested recipe across the curator-on-Spark series:

- **[finance-chat-GGUF](https://huggingface.co/Orionfold/finance-chat-GGUF)** — AdaptLLM finance-chat (Llama-2-7B lineage) for FinanceBench-shaped queries
- **[Saul-7B-Instruct-v1-GGUF](https://huggingface.co/Orionfold/Saul-7B-Instruct-v1-GGUF)** — Equall Saul-7B legal-instruct for LegalBench-shaped queries
- **[zephyr-7b-cyber-GGUF](https://huggingface.co/Orionfold/zephyr-7b-cyber-GGUF)** — Mistral-7B + Zephyr DPO with cyber-eval gating
- **[II-Medical-8B-GGUF](https://huggingface.co/Orionfold/II-Medical-8B-GGUF)** — Qwen3-8B + DAPO reasoning for MedMCQA-shaped queries

Each card lists its own measurement quad; the headline numbers are recorded as the actual sweep ran, never pre-corrected.
```

**Maintenance rule:** every new vertical card edits this block on the previous N cards as part of the push session. Don't ship vertical #5 without back-editing finance/legal/cyber/medical card cross-links to include it. (This is the cross-link half of the engagement-pull lever — without backfill, only the newest card benefits.)

### 3. Wire-back to article + llms.txt

The `## Methods` section already links to the article at `https://ainative.business/field-notes/<slug>/`. **Two additional wire-backs strengthen the loop:**

- **`## Read the deep-dive` block** above the cross-links — explicit invitation to read the article, with one-line abstract pulled from the article's frontmatter
- **`llms.txt` entry** at `https://ainative.business/llms.txt` for this specific card → article slug pairing. When an LLM agent surfaces the model, the llms.txt entry tells it where the canonical methods writeup lives

Check 4 already validates the article exists locally; this reference adds the requirement that the wire-back is *explicit and visible*, not just a buried URL.

### 4. Launch-list call — the engagement endpoint

Engagement-pull without a conversion endpoint is a vanity metric. The Orionfold funnel is: HF card → article deep-dive → orionfold.com mailing list. **The endpoint is the launch list**, not a paid Sponsors page — Orionfold's commercial brand is still pre-launch, asking for sponsorship is premature. Every card includes a launch-list line in the footer:

```markdown
> Want to know when the next Orionfold vertical curator drops? [Join the launch list at orionfold.com](https://orionfold.com).
```

The placement matters: footer, after the publisher attribution line, single line, conversational. Not an ad-block; a "stay in the loop" credit-line that captures interested traffic into a real channel.

**When Sponsors becomes the right endpoint** (revisit per `[[project_orionfold_parent_brand]]`):
- Orionfold has shipped 6+ verticals (commercial credibility floor)
- A working `github.com/sponsors/manavsehgal` page exists (don't link before it exists — a 404 conversion endpoint is worse than nothing)
- The product launch on orionfold.com has happened (sponsorship asks land better post-launch)

Until those conditions hold, the launch list is the right call.

### 5. Frontmatter metadata completeness

The bones. Without these, HF's discoverability surfaces (model search, leaderboards, the `pipeline_tag` filter) won't surface the card regardless of card content:

- `pipeline_tag: text-generation` (or appropriate non-text-generation tag — verify against the model's actual output shape; default is correct for chat-tuned GGUFs)
- `library_name: gguf` (or `transformers` for non-GGUF formats)
- `tags:` — non-empty list of **at least 3** entries including:
  - `spark-tested` (Orionfold differentiator — required)
  - `gguf` (format)
  - `llama-cpp` (runtime hint)
  - `<vertical>` (finance / legal / cyber / medical / patent / …)
  - Optional: chat-format hint (`llama-2`, `chatml`, etc.), license-flavor tag (`apache-2.0`), or HF taxonomy tag (`text-generation-inference`)

`scripts/verify_stage.sh` Check 6 enforces this contract. The `$VERIFY_REQUIRED_TAGS` env var defaults to `spark-tested` and is comma-separated for additional required tags; `$VERIFY_MIN_TAGS` defaults to 3.

**How to populate from publish_quant:** thread a `tags=(...)` tuple kwarg into the call. The `g3_build_first_quant.sh` orchestrator should set this automatically per the resolved vertical; if it isn't doing so today, that's a fieldkit gap and the fix path is either patching the orchestrator or passing the kwarg explicitly in step 2 of the workflow.

## The retro-fix playbook (already-pushed cards)

When `card-audit` mode flags an existing HF card as gap-ridden, the fix is straightforward but session-discipline matters:

1. Pull the current `README.md` from the HF repo to `/tmp/card-audit-<slug>/`.
2. Diff against the desired shape (this reference is the source of truth).
3. Patch in place — add `## Other Orionfold vertical curators`, fix `tags:` frontmatter, add Sponsors footer, add `## Read the deep-dive`.
4. Use `huggingface_hub.upload_file` with `commit_message="Card polish — engagement-pull metadata + cross-links"` to push just the README. **Do not** re-run `publish_quant` — that re-stages and re-uploads all the GGUFs.
5. After push, back-edit the cross-link block on every other Orionfold card (since now there's one more card that should appear in everyone's "Other curators" list).

Per `[[feedback_handoff_md_update_protocol]]`, log the retro-fix in HANDOFF.md and SYNC-HANDOFF.md as part of the session-close.

## When to bend these rules

- **Q5_K_M not present in variants** — recommended_variant must be one of the actual variants. If the quant sweep ran a different mix (e.g., Q4_K_M / Q6_K / Q8_0 only), pick the closest quality-per-byte point and explain in `## Recommendations`.
- **Single-vertical cycles** — the "Other Orionfold vertical curators" block can be omitted on the *very first* card in a series (no siblings yet). From card #2 onward, it's required.
- **Non-GGUF artifacts (LoRA adapters)** — `library_name` becomes `transformers` or `peft`; `pipeline_tag` may shift to `text-classification` etc. depending on adapter intent. The five engagement-pull elements still apply.
- **Non-commercial license cards** — Sponsors line can stay; engagement-pull doesn't depend on commercialization. (Though per `[[project_orionfold_parent_brand]]`, Orionfold's commercial tier prefers permissive licenses; a non-permissive card is a deviation that should be flagged in the scout report.)

## Memory cross-references

- `[[feedback_customer_link_audit]]` — voice-and-style audit (separate from this engagement-pull audit, but pairs with it at push time)
- `[[feedback_handoff_md_update_protocol]]` — the session-close log for retro-fixes
- `[[project_q8_anomaly_model_specific]]` — the "never pre-correct Q8_0" rule applies to the Spark-tested table
- `[[project_orionfold_parent_brand]]` — commercial-tier framing
- `[[feedback_refresh_stats_on_publish]]` — the post-fix HANDOFF + stats refresh tail

## What's intentionally NOT in this reference

- **Article voice + structure** — that's `tech-writer` skill's `references/voice-and-style.md` + `article-structure.md`. The customer-link audit at SKILL.md step 4 cross-references that.
- **Renderer logic** — that's `fieldkit/src/fieldkit/publish/__init__.py`. If the renderer needs to auto-emit cross-link blocks or Sponsors footers, that's a v0.4.x fieldkit change tracked through `fieldkit-curator`.
- **Discoverability A/B testing** — once 6+ verticals have shipped with this recipe and the engagement gap closes (or doesn't), this reference will get a "what moved the needle" pulse section. Until then, the recipe is hypothesis-driven from the 4-vertical pulse.
