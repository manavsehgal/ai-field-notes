"""
A5 — code-edit policy rails for the Autoresearch agent.

Four rails wrap every LLM-proposed perturbation. They are programmatic
(no LLM-as-judge) so the rails themselves cannot drift, hallucinate, or
get prompt-injected. Each rail returns (ok, reason). The pipeline stops
at the first failing rail and logs the reason to the trajectory.

  R1 schema_rail        : input is well-formed JSON with {knob, new_value, reason}
  R2 menu_rail          : knob is in the perturbation_menu.json allowlist
  R3 range_rail         : new_value matches the knob's type AND range/choices
  R4 cross_rail         : applying the knob preserves cross-constraints
                          (d_model % n_head == 0, etc.)
  R5 diff_lint_rail     : the unified diff produced by applying the
                          knob touches exactly one Cfg field, no
                          imports, no Python execution surface
                          (no eval/exec/__import__/subprocess/os.system/etc.)

The rails surface as a single function `gate(proposal_json: str, baseline_cfg: dict) -> Verdict`.

This module has *no Guardrails framework dependency* — it is pure Python
+ ast + regex. NeMo Guardrails wraps it in actions.py for the same
input-rail / output-rail Colang flow A5's sibling articles use. Both
paths run the same checks; the framework wrap is for compatibility with
F7's observability.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVIDENCE = Path(__file__).resolve().parent
MENU_PATH = EVIDENCE / "perturbation_menu.json"


@dataclass
class Verdict:
    ok: bool
    rail: str           # which rail returned this verdict
    reason: str
    proposal: dict[str, Any] | None = None
    diff: str | None = None


def load_menu() -> dict[str, Any]:
    with open(MENU_PATH) as f:
        return json.load(f)


# -------------- R1 schema_rail --------------------------------------------
SCHEMA_REQUIRED_KEYS = {"knob", "new_value", "reason"}


def schema_rail(raw: str) -> tuple[bool, str, dict | None]:
    if not raw or not raw.strip():
        return False, "empty proposal", None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"json parse: {e.msg}", None
    if not isinstance(obj, dict):
        return False, "proposal is not a JSON object", None
    if set(obj.keys()) != SCHEMA_REQUIRED_KEYS:
        extra = set(obj.keys()) - SCHEMA_REQUIRED_KEYS
        missing = SCHEMA_REQUIRED_KEYS - set(obj.keys())
        return False, f"schema mismatch (extra={extra}, missing={missing})", None
    if not isinstance(obj["knob"], str):
        return False, "knob must be string", None
    if not isinstance(obj["reason"], str) or len(obj["reason"]) > 500:
        return False, "reason must be string ≤ 500 chars", None
    return True, "ok", obj


# -------------- R2 menu_rail ----------------------------------------------
def menu_rail(proposal: dict, menu: dict) -> tuple[bool, str]:
    knob = proposal["knob"]
    if knob not in menu["knobs"]:
        return False, f"knob '{knob}' not in allowlist (knobs={list(menu['knobs'].keys())})"
    return True, "ok"


# -------------- R3 range_rail ---------------------------------------------
def range_rail(proposal: dict, menu: dict) -> tuple[bool, str]:
    knob = proposal["knob"]
    spec = menu["knobs"][knob]
    val = proposal["new_value"]

    expected_type = {"int": int, "float": (int, float), "str": str}[spec["type"]]
    if not isinstance(val, expected_type):
        return False, (f"new_value type mismatch: knob '{knob}' expects {spec['type']}, "
                       f"got {type(val).__name__}")
    if spec["type"] == "int" and isinstance(val, bool):
        return False, "boolean is not a valid int"

    if "range" in spec:
        lo, hi = spec["range"]
        if val < lo or val > hi:
            return False, f"new_value {val} outside range [{lo}, {hi}] for knob '{knob}'"
    if "choices" in spec:
        if val not in spec["choices"]:
            return False, f"new_value {val} not in choices {spec['choices']} for knob '{knob}'"
    return True, "ok"


# -------------- R4 cross_rail ---------------------------------------------
def cross_rail(proposal: dict, baseline_cfg: dict, menu: dict) -> tuple[bool, str]:
    """Apply the proposal to the baseline cfg and check every cross-
    constraint. We evaluate each rule string in a tiny restricted
    namespace — only the cfg dict's keys, no builtins."""
    new_cfg = dict(baseline_cfg)
    new_cfg[proposal["knob"]] = proposal["new_value"]
    safe_globals = {"__builtins__": {}}
    for c in menu.get("cross_constraints", []):
        rule = c["rule"]
        # Whitelist: rule may only reference keys in new_cfg + arithmetic.
        # ast walk to enforce.
        try:
            tree = ast.parse(rule, mode="eval")
        except SyntaxError as e:
            return False, f"cross-constraint '{c['id']}' has invalid syntax: {e.msg}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in new_cfg:
                    return False, (f"cross-constraint '{c['id']}' references unknown "
                                   f"variable '{node.id}'")
            elif isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
                return False, (f"cross-constraint '{c['id']}' uses disallowed "
                               f"node type {type(node).__name__}")
        try:
            ok = bool(eval(compile(tree, "<rule>", "eval"), safe_globals, new_cfg))
        except Exception as e:  # noqa: BLE001
            return False, f"cross-constraint '{c['id']}' raised at eval: {e}"
        if not ok:
            return False, (f"cross-constraint '{c['id']}' violated: {rule} "
                           f"(applied cfg: {new_cfg})")
    return True, "ok"


# -------------- R5 diff_lint_rail -----------------------------------------
DISALLOWED_TOKENS = re.compile(
    r"\b(eval|exec|compile|__import__|globals|locals|"
    r"subprocess|os\.system|os\.popen|open|"
    r"socket|requests|urllib|httpx|"
    r"pickle|marshal|"
    r"setattr|delattr|getattr)\b"
)


def make_diff(baseline_cfg: dict, proposal: dict) -> str:
    """Construct the canonical 1-line cfg-mutation diff for this proposal.
    A real apply would patch a Cfg dataclass field assignment; this diff
    is the textual artifact the lint rail checks."""
    knob = proposal["knob"]
    old = baseline_cfg.get(knob)
    new = proposal["new_value"]
    return (
        f"--- a/cfg.py\n"
        f"+++ b/cfg.py\n"
        f"@@ -1 +1 @@\n"
        f"-    {knob}: {type(old).__name__} = {old!r}\n"
        f"+    {knob}: {type(new).__name__} = {new!r}\n"
    )


def diff_lint_rail(diff: str, proposal: dict) -> tuple[bool, str]:
    """The diff must contain exactly one '+' line and one '-' line in
    the body, and the changed line must touch only the proposed knob.
    No disallowed tokens anywhere."""
    plus_lines = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    minus_lines = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    if len(plus_lines) != 1 or len(minus_lines) != 1:
        return False, (f"diff must have exactly one '+' and one '-' body line, "
                       f"got +{len(plus_lines)}/-{len(minus_lines)}")
    body = "\n".join(plus_lines + minus_lines)
    m = DISALLOWED_TOKENS.search(body)
    if m:
        return False, f"disallowed token in diff: '{m.group(0)}'"
    if proposal["knob"] not in plus_lines[0]:
        return False, f"diff '+' line does not mention knob '{proposal['knob']}'"
    return True, "ok"


# -------------- top-level gate --------------------------------------------
def gate(raw: str, baseline_cfg: dict, menu: dict | None = None) -> Verdict:
    if menu is None:
        menu = load_menu()

    ok, reason, proposal = schema_rail(raw)
    if not ok:
        return Verdict(ok=False, rail="R1_schema", reason=reason, proposal=None)

    ok, reason = menu_rail(proposal, menu)
    if not ok:
        return Verdict(ok=False, rail="R2_menu", reason=reason, proposal=proposal)

    ok, reason = range_rail(proposal, menu)
    if not ok:
        return Verdict(ok=False, rail="R3_range", reason=reason, proposal=proposal)

    ok, reason = cross_rail(proposal, baseline_cfg, menu)
    if not ok:
        return Verdict(ok=False, rail="R4_cross", reason=reason, proposal=proposal)

    diff = make_diff(baseline_cfg, proposal)
    ok, reason = diff_lint_rail(diff, proposal)
    if not ok:
        return Verdict(ok=False, rail="R5_diff_lint", reason=reason,
                       proposal=proposal, diff=diff)

    return Verdict(ok=True, rail="passed", reason="all rails passed",
                   proposal=proposal, diff=diff)
