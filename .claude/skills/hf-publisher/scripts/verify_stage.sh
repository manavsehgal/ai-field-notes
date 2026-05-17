#!/usr/bin/env bash
# hf-publisher — automated stage verification before live HF push.
#
# Catches the same five rendering bugs that bit `Orionfold/finance-chat-GGUF`:
# wrong license-frontmatter default, empty `## How to run` body, mis-shaped
# Spark-tested table, broken Methods link, GGUF files in stage that the
# Variants table doesn't list. Each is a reason a customer-facing card would
# render wrong on HuggingFace and we'd find out only after the push.
#
# Usage:  bash verify_stage.sh /tmp/orionfold-stage/<slug>
# Exit code = number of failed checks. 0 = ready to push.

set -uo pipefail

STAGE_DIR="${1:-}"
if [[ -z "$STAGE_DIR" ]]; then
  echo "Usage: $0 <stage-dir>" >&2
  exit 99
fi
if [[ ! -d "$STAGE_DIR" ]]; then
  echo "ERROR: stage dir does not exist: $STAGE_DIR" >&2
  exit 99
fi
README="$STAGE_DIR/README.md"
if [[ ! -f "$README" ]]; then
  echo "ERROR: $README missing — run dry-run first" >&2
  exit 99
fi

# Articles dir is canonical at this path. Override via ARTICLES_DIR if needed.
ARTICLES_DIR="${ARTICLES_DIR:-/home/nvidia/ai-field-notes/articles}"

PASS=0
FAIL=0

pass() { printf "[\033[1;32mPASS\033[0m] %s\n" "$*"; PASS=$((PASS+1)); }
fail() { printf "[\033[1;31mFAIL\033[0m] %s\n" "$*"; FAIL=$((FAIL+1)); }

# --- Check 1: license frontmatter ----------------------------------------
# The bug we shipped on finance-chat: license defaulted to apache-2.0 but the
# model is Llama-2 lineage. Allow apache-2.0 ONLY if the source repo's README
# explicitly says Apache (caller verified) — flagged via APACHE_VERIFIED=1.
license_line=$(awk '/^---$/{f=!f; next} f && /^license:/' "$README" | head -1)
license_value=$(echo "$license_line" | sed 's/^license:[[:space:]]*//' | tr -d '"' | tr -d "'")
if [[ -z "$license_value" ]]; then
  fail "license frontmatter is missing entirely"
elif [[ "$license_value" == "apache-2.0" ]]; then
  if [[ "${APACHE_VERIFIED:-0}" == "1" ]]; then
    pass "license frontmatter is apache-2.0 (caller-verified upstream is Apache)"
  else
    fail "license: apache-2.0 — verify upstream isn't Llama/Gemma/Qwen/CC-BY-NC; pass APACHE_VERIFIED=1 if confirmed"
  fi
else
  pass "license frontmatter is non-default (got: $license_value)"
fi

# --- Check 2: ## How to run body is non-empty ----------------------------
# The bug we shipped on finance-chat: section header rendered with no body
# because ollama_pull_handle + transformers_snippet were both None and the
# old renderer had no GGUF default. Threshold: ≥ 8 non-empty content lines
# between the header and the next ## heading (covers minimal pull + serve
# snippet pair).
howto_body=$(awk '
  /^## How to run/ { in_section=1; next }
  in_section && /^## / { exit }
  in_section { print }
' "$README" | grep -cE '\S')
if (( howto_body >= 8 )); then
  pass "## How to run body is non-empty ($howto_body content lines)"
else
  fail "## How to run body has only $howto_body content lines (need ≥ 8) — likely the empty-section bug"
fi

# --- Check 3: ## Spark-tested table shape --------------------------------
# Columns are *metrics* (Variant / Size / Perplexity / tok-s / optional
# vertical-eval), not derived from variants count. Valid shapes:
#   - 4 cols (no vertical eval): Variant | Size | Perplexity | tok/s
#   - 5 cols (with vertical eval): Variant | Size | Perplexity | tok/s | <eval>
# Each VARIANT-ROW (excluding header + separator) must match the header
# column count — that's the real shape invariant.
sparktable_header=$(awk '
  /^## Spark-tested/ { in_section=1; next }
  in_section && /^## / { exit }
  in_section && /^\| Variant \|/ { print; exit }
' "$README")
if [[ -z "$sparktable_header" ]]; then
  fail "## Spark-tested table is missing"
else
  header_pipes=$(echo "$sparktable_header" | tr -cd '|' | wc -c)
  header_cells=$((header_pipes - 1))
  # Pull all data rows (rows after the |---| separator inside Spark-tested)
  rows_with_wrong_cell_count=$(awk -v hcells="$header_cells" '
    /^## Spark-tested/ { in_section=1; next }
    in_section && /^## / { exit }
    in_section && /^\| / && !/^\| Variant \|/ && !/^\|---/ {
      pipes = gsub(/\|/, "|")
      cells = pipes - 1
      if (cells != hcells) { bad++ }
    }
    END { print bad+0 }
  ' "$README")
  if (( header_cells != 4 && header_cells != 5 )); then
    fail "Spark-tested header has $header_cells cells; expected 4 (no vertical-eval) or 5 (with vertical-eval)"
  elif (( rows_with_wrong_cell_count > 0 )); then
    fail "Spark-tested table has $rows_with_wrong_cell_count data row(s) whose cell count != header ($header_cells)"
  else
    pass "Spark-tested table shape is correct ($header_cells cols, all rows match)"
  fi
fi

# --- Check 4: ## Methods link points at existing article -----------------
methods_url=$(awk '
  /^## Methods/ { in_section=1; next }
  in_section && /^## / { exit }
  in_section { print }
' "$README" | grep -oE 'ainative\.business/field-notes/[a-z0-9-]+/?' | head -1)
if [[ -z "$methods_url" ]]; then
  fail "## Methods link is missing or malformed"
else
  slug=$(echo "$methods_url" | sed -E 's|.*/field-notes/([a-z0-9-]+)/?|\1|')
  if [[ -d "$ARTICLES_DIR/$slug" ]]; then
    pass "Methods link points at existing article ($slug)"
  else
    fail "Methods link slug '$slug' has no $ARTICLES_DIR/$slug directory"
  fi
fi

# --- Check 5: Variants table covers every model-*.gguf in stage ----------
table_variants=$(awk '
  /^## Variants/ { in_section=1; next }
  in_section && /^## / { exit }
  in_section && /^\| / && !/^\| Variant \|/ && !/^\|---/ { print $2 }
' "$README" | tr -d ' ')
stage_variants=$(ls "$STAGE_DIR" | grep -oE '^model-[A-Za-z0-9_]+\.gguf$' | sed -E 's/^model-(.+)\.gguf$/\1/')
if [[ -z "$stage_variants" ]]; then
  fail "no model-*.gguf files in stage — nothing to publish"
else
  missing=()
  while IFS= read -r v; do
    [[ -z "$v" ]] && continue
    if ! grep -qx "$v" <<< "$table_variants"; then
      missing+=("$v")
    fi
  done <<< "$stage_variants"
  if (( ${#missing[@]} == 0 )); then
    n=$(echo "$stage_variants" | wc -l)
    pass "Variants table covers all $n GGUF files in stage"
  else
    fail "Variants table missing rows for: ${missing[*]}"
  fi
fi

# --- Summary --------------------------------------------------------------
TOTAL=$((PASS + FAIL))
if (( FAIL == 0 )); then
  printf "\n\033[1;32m%d/%d PASSED\033[0m — stage is ready for live push\n" "$PASS" "$TOTAL"
else
  printf "\n\033[1;31m%d/%d FAILED\033[0m — fix before pushing\n" "$FAIL" "$TOTAL"
fi
exit "$FAIL"
