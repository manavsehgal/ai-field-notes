"""T²PO rollout driver — TDS regeneration + thinking-token cap.

Wraps the Phase 6 RolloutDriver logic with two T²PO-specific deltas:

  1. **Token-level intervention — `num_think_tokens` cap.**
     The paper's marginal-uncertainty trigger collapses to a hard
     budget on the response length. We forward `num_think_tokens` as
     `max_tokens` on every vLLM generate call. Default 450.

  2. **Turn-level intervention — TDS (Trajectory-level Distillation
     Sampling) regenerate loop.**
     After each turn's generation, compute the mean per-token entropy
     of the response (from vLLM's top-k logprobs). If `_step > 0` and
     the absolute change from the previous turn's entropy lies in
     `(0, eta_threshold)` (default 0.3), the turn was an "uncertain
     plateau" — resample the turn (up to `max_try=2` times) before
     committing it to the trajectory. The rationale (per the paper):
     these plateau turns are where the policy is most likely to lock
     into a degenerate continuation; resampling reroutes the
     trajectory through a different action.

The rollout re-uses `rollout.py`'s `parse_action`, `LocalTempSandbox`,
`Trajectory`, `TurnRecord`, and prompt templates — no logic forks.

vLLM logprobs caveat. vLLM's OpenAI-compatible chat endpoint returns
top-k logprobs (not the full vocab distribution). We approximate
per-token entropy from the top-k distribution after re-normalizing it
to sum to 1. Top-20 captures most of the probability mass for the
sharply-peaked distributions typical in agentic generation; it under-
estimates entropy slightly when the policy is genuinely uncertain
(the tail mass is dropped) — but the relative comparison across turns
that TDS keys on is preserved.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import requests

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from rollout import (  # noqa: E402
    LocalTempSandbox,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    Sandbox,
    Trajectory,
    TurnRecord,
    parse_action,
    render_files_block,
    truncate,
)


def chat_with_logprobs(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    top_logprobs: int = 20,
    timeout: float = 180.0,
) -> tuple[str, float]:
    """Single chat call → (response_text, mean_per_token_entropy).

    Uses the vLLM OpenAI-compatible endpoint with `logprobs=True` and
    `top_logprobs=K` to recover an approximate per-token distribution.
    Entropy = -Σ p_k log p_k over the top-K (re-normalized) probs.
    Returns 0.0 entropy if the server didn't return logprobs (e.g.,
    empty response).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    text = choice["message"]["content"]
    lp = choice.get("logprobs") or {}
    content_lps = lp.get("content") or []
    if not content_lps:
        return text, 0.0

    entropies: list[float] = []
    for tok in content_lps:
        top = tok.get("top_logprobs") or []
        if not top:
            continue
        # logprob → unnormalized prob → re-normalize over the top-K we got back
        raw_logp = [t["logprob"] for t in top]
        m = max(raw_logp)
        unnorm = [math.exp(lp - m) for lp in raw_logp]
        z = sum(unnorm)
        if z <= 0:
            continue
        probs = [p / z for p in unnorm]
        h = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        entropies.append(h)
    if not entropies:
        return text, 0.0
    return text, sum(entropies) / len(entropies)


class T2PORolloutDriver:
    """Drop-in replacement for RolloutDriver with TDS + CoT cap."""

    def __init__(
        self,
        *,
        vllm_base_url: str,
        model_name: str,
        sandbox_factory,
        max_turns: int = 12,
        per_command_timeout: float = 10.0,
        debug: bool = False,
        temperature: float = 0.8,
        num_think_tokens: int = 450,
        tds_eta_threshold: float = 0.3,
        tds_max_try: int = 2,
        top_logprobs: int = 20,
        chat_timeout: float = 180.0,
    ) -> None:
        self.vllm_base_url = vllm_base_url
        self.model_name = model_name
        self.sandbox_factory = sandbox_factory
        self.max_turns = max_turns
        self.per_command_timeout = per_command_timeout
        self.debug = debug
        self.temperature = temperature
        self.num_think_tokens = num_think_tokens
        self.tds_eta_threshold = tds_eta_threshold
        self.tds_max_try = tds_max_try
        self.top_logprobs = top_logprobs
        self.chat_timeout = chat_timeout

    def _generate_turn(self, messages: list[dict]) -> tuple[str, float, int]:
        """Generate one assistant turn, with TDS resampling.

        Returns (response_text, entropy_used, n_attempts). The first
        sample is always taken; resampling only fires when the prior
        turn's entropy is set (i.e., the caller passes a non-None
        prev_entropy via self._prev_entropy) AND the change falls in
        (0, eta_threshold).
        """
        text, h = chat_with_logprobs(
            self.vllm_base_url,
            self.model_name,
            messages,
            temperature=self.temperature,
            max_tokens=self.num_think_tokens,
            top_logprobs=self.top_logprobs,
            timeout=self.chat_timeout,
        )
        attempts = 1
        prev = getattr(self, "_prev_entropy", None)
        if prev is None:
            return text, h, attempts
        for _ in range(self.tds_max_try):
            delta = abs(h - prev)
            if delta <= 0 or delta >= self.tds_eta_threshold:
                break
            text, h = chat_with_logprobs(
                self.vllm_base_url,
                self.model_name,
                messages,
                temperature=self.temperature,
                max_tokens=self.num_think_tokens,
                top_logprobs=self.top_logprobs,
                timeout=self.chat_timeout,
            )
            attempts += 1
        return text, h, attempts

    def rollout(self, task: dict) -> tuple[Trajectory, Sandbox]:
        sandbox = self.sandbox_factory()
        sandbox.materialize(task)
        t0 = time.time()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    intent=task["intent"],
                    file_listing=render_files_block(sandbox.list_files()),
                ),
            },
        ]

        turns: list[TurnRecord] = []
        stopped = "max_turns"
        self._prev_entropy = None
        total_regen_attempts = 0
        for turn_idx in range(1, self.max_turns + 1):
            try:
                response_text, turn_entropy, attempts = self._generate_turn(messages)
            except Exception as e:
                if self.debug:
                    print(f"  turn {turn_idx}: agent error: {e}", file=sys.stderr)
                turns.append(TurnRecord(
                    turn=turn_idx,
                    agent_response="",
                    action=None,
                    observation=None,
                    parse_error=f"agent_error: {type(e).__name__}: {e}",
                ))
                stopped = "agent_error"
                break

            total_regen_attempts += (attempts - 1)
            self._prev_entropy = turn_entropy
            action, parse_error = parse_action(response_text)

            if self.debug:
                print(f"  turn {turn_idx}: H={turn_entropy:.3f} attempts={attempts} "
                      f"resp[:120]={response_text[:120]!r}", file=sys.stderr)

            if action is None:
                turns.append(TurnRecord(
                    turn=turn_idx,
                    agent_response=truncate(response_text),
                    action=None,
                    observation=None,
                    parse_error=parse_error,
                ))
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"PARSE ERROR: {parse_error}. Reply with ONE ```bash``` block "
                        "containing one command, or TASK_COMPLETE on a line by itself."
                    ),
                })
                continue

            if action["kind"] == "done":
                turns.append(TurnRecord(
                    turn=turn_idx,
                    agent_response=truncate(response_text),
                    action=action,
                    observation=None,
                ))
                stopped = "task_complete"
                break

            obs = sandbox.exec(action["cmd"], timeout=self.per_command_timeout)
            obs_dict = obs.to_dict()
            obs_dict["stdout"] = truncate(obs_dict["stdout"])
            obs_dict["stderr"] = truncate(obs_dict["stderr"])

            turns.append(TurnRecord(
                turn=turn_idx,
                agent_response=truncate(response_text),
                action=action,
                observation=obs_dict,
            ))

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    f"OBSERVATION (exit {obs.exit_code}{', TIMED OUT' if obs.timed_out else ''}):\n"
                    f"--- stdout ---\n{obs_dict['stdout']}\n"
                    f"--- stderr ---\n{obs_dict['stderr']}\n"
                    "Next command (one ```bash``` block) or TASK_COMPLETE."
                ),
            })

        wall = time.time() - t0
        traj = Trajectory(
            task_id=task["task_id"],
            model=self.model_name,
            n_turns=len(turns),
            stopped=stopped,
            wall_seconds=wall,
            turns=turns,
        )
        traj.tds_regen_attempts = total_regen_attempts  # type: ignore[attr-defined]
        return traj, sandbox
