"""
NeMo Guardrails actions that delegate to rails.py.

Every action is pure Python — no LLM-as-judge, no external service.
That's the whole point of A5: for *agent-action policy* (vs. user-input
policy), the rails should be programmatic, not LLM-based.
"""
from __future__ import annotations

import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVIDENCE))
from rails import (  # noqa: E402
    schema_rail, menu_rail, range_rail, cross_rail, diff_lint_rail,
    make_diff, load_menu,
)
from nemoguardrails.actions import action  # noqa: E402

BASELINE_CFG = {
    "n_layer": 24, "n_head": 16, "d_model": 1024, "d_ff": 4096,
    "lr": 3e-4, "lr_warmup": 5, "grad_clip": 1.0, "weight_decay": 0.0,
    "beta1": 0.9, "beta2": 0.95,
    "batch_size": 16, "seq_len": 1024, "precision": "fp8",
}


@action(name="rails_schema")
async def _schema(text: str):
    ok, reason, _ = schema_rail(text)
    return {"blocked": not ok, "rail": "R1_schema", "reason": reason}


@action(name="rails_menu_and_range")
async def _menu_and_range(text: str):
    menu = load_menu()
    ok, reason, proposal = schema_rail(text)
    if not ok:
        return {"blocked": True, "rail": "R1_schema", "reason": reason}
    ok, reason = menu_rail(proposal, menu)
    if not ok:
        return {"blocked": True, "rail": "R2_menu", "reason": reason}
    ok, reason = range_rail(proposal, menu)
    if not ok:
        return {"blocked": True, "rail": "R3_range", "reason": reason}
    return {"blocked": False, "rail": "ok", "reason": "menu+range pass"}


@action(name="rails_cross")
async def _cross(text: str):
    menu = load_menu()
    ok, reason, proposal = schema_rail(text)
    if not ok:
        return {"blocked": True, "rail": "R1_schema", "reason": reason}
    ok, reason = cross_rail(proposal, BASELINE_CFG, menu)
    if not ok:
        return {"blocked": True, "rail": "R4_cross", "reason": reason}
    return {"blocked": False, "rail": "ok", "reason": "cross pass"}


@action(name="rails_diff_lint")
async def _diff_lint(text: str):
    menu = load_menu()
    ok, reason, proposal = schema_rail(text)
    if not ok:
        return {"blocked": True, "rail": "R1_schema", "reason": reason}
    diff = make_diff(BASELINE_CFG, proposal)
    ok, reason = diff_lint_rail(diff, proposal)
    if not ok:
        return {"blocked": True, "rail": "R5_diff_lint", "reason": reason}
    return {"blocked": False, "rail": "ok", "reason": "diff lint pass"}
