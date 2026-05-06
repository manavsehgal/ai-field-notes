# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""Evaluation primitives lifted from the project's eval-shaped articles.

Three building blocks:

- `Bench` — wall-clock + per-call metric collector with mean/median/min/max
  aggregation. Drop-in replacement for the hand-rolled `benchmark.py` files
  under `articles/*/evidence/`.
- `Judge` — LLM-as-judge wrapper around `fieldkit.nim.NIMClient`. Built-in
  rubrics for correctness (0-5), faithfulness (0-1), and relevance (0-1)
  match `articles/rag-eval-ragas-and-nemo-evaluator/evidence/grade.py` and
  `articles/lora-on-your-own-qa-pairs/evidence/judge.py` verbatim.
- `Trajectory` — analyzer for the agent-loop JSONL the
  `autoresearch-agent-loop` article emits. Computes knob coverage,
  repeat rate (with optional sliding window), mode dominance, and
  cumulative-best curves. Schema documented inline.

Plus the top-level `is_refusal(text) -> bool` regex helper, since two
articles independently use it for refusal-rate accounting.

`Bench` is offline-only: it just measures wall time around a callable
and aggregates whatever numeric fields the callable returns. `Judge`
needs a warm NIM (or any chat-completions endpoint via `NIMClient`).
`Trajectory` is a pure file parser — no network.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fieldkit.nim import NIMClient, NIMError

__all__ = [
    "AssertionGrader",
    "AssertionResult",
    "Bench",
    "BenchCall",
    "GradeResult",
    "Judge",
    "JudgeError",
    "JudgeResult",
    "Trajectory",
    "TrajectoryIter",
    "RUBRIC_CORRECTNESS",
    "RUBRIC_FAITHFULNESS",
    "RUBRIC_RELEVANCE",
    "BUILTIN_RUBRICS",
    "REFUSAL_PATTERNS",
    "ASSERTION_KINDS",
    "is_refusal",
    "summarize_metric",
]


# --- Refusal detector ----------------------------------------------------

REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"i (?:do not|don['’]t) (?:know|have)",
        r"i (?:cannot|can['’]t|am (?:not )?able to) (?:answer|provide|determine|find)",
        r"(?:the )?(?:provided )?context (?:does not|doesn['’]t) (?:contain|include|mention)",
        r"not (?:specified|mentioned|provided|available|stated|given|directly|explicitly)",
        r"i (?:am unable|cannot) to (?:determine|verify)",
        r"no (?:specific|direct)? ?information",
        r"\bunclear\b",
        r"insufficient (?:information|context|data)",
        r"cannot be determined",
    )
)
"""Union of the refusal regex catalogs from `rag-eval-ragas-and-nemo-evaluator`
and `lora-on-your-own-qa-pairs`. Compiled case-insensitive."""


def is_refusal(text: str | None) -> bool:
    """Return True if `text` looks like a model refusal.

    Empty / None input counts as a refusal so refusal rates stay
    well-defined on missing predictions. The pattern union is broad on
    purpose: false-positive refusal flags are far less harmful than
    silently counting a hedged non-answer as a real answer.
    """
    if not text:
        return True
    return any(p.search(text) for p in REFUSAL_PATTERNS)


# --- Bench ---------------------------------------------------------------


def summarize_metric(values: Iterable[float | None]) -> dict[str, float | int]:
    """Mean / median / min / max for a metric series, ignoring None.

    Matches the shape used by the project's hand-rolled `benchmark.json`
    files so a side-by-side diff with article evidence stays meaningful.
    Returns ``{"n": 0}`` for an empty (or all-None) series.
    """
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": round(sum(vals) / n, 2),
        "median": vals[n // 2],
        "min": vals[0],
        "max": vals[-1],
    }


@dataclass
class BenchCall:
    """One entry in `Bench.calls`. Fully serializable via `asdict()`."""

    input: Any
    output: Any
    latency_ms: float
    success: bool = True
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class Bench:
    """Wall-clock benchmark harness with numeric metric aggregation.

    Usage::

        def slow_double(x):
            return {"value": x * 2, "tokens": x}

        with Bench("doubler", metrics=["tokens"]) as b:
            b.run(slow_double, [1, 2, 3])
            print(b.report())
            b.dump("bench.json")

    The callable supplied to `.run()` receives one input at a time and may
    return any value. If it returns a dict, entries whose keys appear in
    `metrics` (top-level) or under `metrics_key` (one nested dict, like
    the project's `timings_ms` shape) are collected for aggregation.

    Exceptions raised by the callable are caught and recorded with
    `success=False` so a single bad input doesn't sink the rest of the
    sweep. Pass `on_error="raise"` to abort on the first failure.

    The context-manager protocol is just for measuring total wall time —
    `with Bench(...)` is purely for aesthetics, not resource cleanup.
    """

    name: str
    metrics: list[str] = field(default_factory=list)
    metrics_key: str | None = None
    """If set, metrics are pulled from ``output[metrics_key]`` rather than
    from the top-level dict. Mirrors the ``timings_ms`` nested shape used
    by the project's `articles/*/evidence/benchmark.py` files."""
    calls: list[BenchCall] = field(default_factory=list)
    wall_seconds: float = 0.0
    _start: float | None = None

    def __enter__(self) -> Bench:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._start is not None:
            self.wall_seconds = time.perf_counter() - self._start

    def run(
        self,
        fn: Callable[[Any], Any],
        inputs: Iterable[Any],
        *,
        on_error: str = "record",
        tag_fn: Callable[[Any], dict[str, Any]] | None = None,
    ) -> Bench:
        """Time `fn(input)` per input; record latency + extracted metrics.

        Returns self so calls can chain. `on_error` is ``"record"`` (the
        default — capture the exception and continue) or ``"raise"`` (let
        the first exception abort the sweep). `tag_fn(input) -> dict` lets
        callers attach metadata to each call without modifying the
        callable.
        """
        if on_error not in ("record", "raise"):
            raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")
        for inp in inputs:
            tags = tag_fn(inp) if tag_fn else {}
            t0 = time.perf_counter()
            try:
                out = fn(inp)
            except Exception as exc:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                if on_error == "raise":
                    raise
                self.calls.append(
                    BenchCall(
                        input=inp,
                        output=None,
                        latency_ms=latency_ms,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        tags=tags,
                    )
                )
                continue
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self.calls.append(
                BenchCall(
                    input=inp,
                    output=out,
                    latency_ms=latency_ms,
                    success=True,
                    metrics=self._extract_metrics(out),
                    tags=tags,
                )
            )
        return self

    def record(
        self,
        *,
        input: Any = None,
        output: Any = None,
        latency_ms: float,
        success: bool = True,
        error: str | None = None,
        tags: dict[str, Any] | None = None,
        **metrics: float,
    ) -> BenchCall:
        """Imperative variant for cases where the caller times its own work.

        Useful when the wrapped function already returns its own latency
        breakdown (embed/retrieve/generate) and you want to record the
        components without re-timing the wall clock.
        """
        call = BenchCall(
            input=input,
            output=output,
            latency_ms=latency_ms,
            success=success,
            error=error,
            metrics=dict(metrics),
            tags=tags or {},
        )
        self.calls.append(call)
        return call

    def _extract_metrics(self, output: Any) -> dict[str, float]:
        if not self.metrics:
            return {}
        if self.metrics_key:
            payload = (
                output.get(self.metrics_key, {})
                if isinstance(output, dict)
                else {}
            )
        else:
            payload = output if isinstance(output, dict) else {}
        if not isinstance(payload, dict):
            return {}
        out: dict[str, float] = {}
        for k in self.metrics:
            v = payload.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(v)
        return out

    def summary(self) -> dict[str, Any]:
        """Aggregate `latency_ms` (always) + each declared metric.

        Plus call counts: `n`, `n_success`, `n_failure`. An empty calls
        list returns ``{"name": ..., "n": 0}``.
        """
        n = len(self.calls)
        if n == 0:
            return {"name": self.name, "n": 0}
        succ = [c for c in self.calls if c.success]
        latencies = [c.latency_ms for c in succ]
        out: dict[str, Any] = {
            "name": self.name,
            "n": n,
            "n_success": len(succ),
            "n_failure": n - len(succ),
            "latency_ms": summarize_metric(latencies),
        }
        for k in self.metrics:
            out[k] = summarize_metric(c.metrics.get(k) for c in succ)
        if self.wall_seconds:
            out["wall_seconds"] = round(self.wall_seconds, 3)
        return out

    def report(self) -> str:
        """Markdown summary table.

        Two-column metric / mean median min max layout. Failed calls
        and the wall-clock total appear as bullets above the table.
        """
        s = self.summary()
        lines: list[str] = [f"### Bench: {s['name']} (n={s.get('n', 0)})"]
        if s.get("n_failure"):
            lines.append(
                f"- success: {s['n_success']} / {s['n']}, failures: {s['n_failure']}"
            )
        if s.get("wall_seconds"):
            lines.append(f"- wall: {s['wall_seconds']}s")
        lines.append("")
        lines.append("| metric | mean | median | min | max |")
        lines.append("|---|---:|---:|---:|---:|")
        for key in ["latency_ms", *self.metrics]:
            cell = s.get(key) or {"n": 0}
            if cell.get("n", 0) == 0:
                lines.append(f"| {key} | — | — | — | — |")
            else:
                lines.append(
                    f"| {key} | {cell['mean']} | {cell['median']} | {cell['min']} | {cell['max']} |"
                )
        return "\n".join(lines)

    def to_dict(self, *, include_outputs: bool = False) -> dict[str, Any]:
        """Serializable dump of summary + per-call records.

        `include_outputs=False` (the default) drops the `output` field
        from each call; outputs are often big LLM payloads not meant
        for archival benchmark.json files. Set True to include them.
        """
        calls: list[dict[str, Any]] = []
        for c in self.calls:
            d = asdict(c)
            if not include_outputs:
                d.pop("output", None)
            calls.append(d)
        return {"summary": self.summary(), "calls": calls}

    def dump(self, path: str | Path, *, include_outputs: bool = False) -> Path:
        """Write `to_dict()` as pretty JSON; returns the path."""
        p = Path(path)
        p.write_text(
            json.dumps(
                self.to_dict(include_outputs=include_outputs),
                indent=2,
                default=_json_default,
            )
        )
        return p


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for sets and arbitrary objects."""
    if isinstance(o, set):
        return list(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


# --- Judge ---------------------------------------------------------------


RUBRIC_CORRECTNESS: str = (
    "You are an impartial grader. Score a predicted answer against a reference "
    "answer on a 0-5 scale: 5=exactly correct, 4=essentially correct with "
    "minor wording differences, 3=partially correct, 2=mostly wrong but with "
    "a correct fragment, 1=confidently wrong or unrelated, 0=refusal or empty. "
    'Return ONLY a JSON object: {"score": N, "rationale": "..."}'
)
"""0-5 correctness rubric, lifted verbatim from
`articles/rag-eval-ragas-and-nemo-evaluator/evidence/grade.py`."""

RUBRIC_FAITHFULNESS: str = (
    "You are an impartial grader of answer faithfulness. Given context passages "
    "and an answer, decide if every factual claim in the answer is SUPPORTED by "
    "the context. Score: 1.0 = all claims supported, 0.5 = some claims "
    "supported, 0.0 = no claims supported. If the answer is a refusal or says "
    "the context doesn't contain the answer, score N/A -> use 0.5 if the "
    "context indeed does not contain a direct answer and 0.0 if it does. "
    'Return ONLY a JSON object: {"score": X, "rationale": "..."}'
)
"""0-1 faithfulness rubric (claims supported by context). Same source."""

RUBRIC_RELEVANCE: str = (
    "You are an impartial grader of answer relevance. Given a question and an "
    "answer, score whether the answer addresses the question: 1.0 = directly "
    "answers, 0.5 = partially addresses or hedges, 0.0 = off-topic or refuses. "
    'Return ONLY a JSON object: {"score": X, "rationale": "..."}'
)
"""0-1 answer-relevance rubric. Same source."""

BUILTIN_RUBRICS: dict[str, str] = {
    "correctness": RUBRIC_CORRECTNESS,
    "faithfulness": RUBRIC_FAITHFULNESS,
    "relevance": RUBRIC_RELEVANCE,
}

_SCORE_RE_TIGHT = re.compile(
    r"\{[^{}]*\"score\"\s*:\s*(?P<score>-?\d+(?:\.\d+)?)[^{}]*\}", re.DOTALL
)
_SCORE_RE_LOOSE = re.compile(r"\"score\"\s*:\s*(?P<score>-?\d+(?:\.\d+)?)")


class JudgeError(Exception):
    """Raised when a judge call fails or returns no parseable score."""


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """One graded prediction. `score` is None iff parsing failed.

    `raw` is the verbatim assistant content; `rationale` is the parsed
    rationale field (or the raw text when JSON parsing failed and we
    fell back to a regex score extraction).
    """

    score: float | None
    rationale: str
    raw: str


@dataclass
class Judge:
    """LLM-as-judge wrapper around a `fieldkit.nim.NIMClient`.

    Built-in rubrics live in `BUILTIN_RUBRICS` (also exported as the
    `RUBRIC_CORRECTNESS`, `RUBRIC_FAITHFULNESS`, `RUBRIC_RELEVANCE`
    constants). Pass any of these as `rubric=...`, or pass a fresh
    string for a custom system prompt — the contract is that the
    judge return a JSON object containing a numeric ``"score"`` field.

    Calls go through `client.chat()` so all of `fieldkit.nim`'s
    guarantees apply: retries on 429/503 + the 8192-token preflight
    that prevents opaque NIM 400s on long context windows.
    """

    client: NIMClient
    rubric: str = RUBRIC_CORRECTNESS
    max_tokens: int = 160
    temperature: float = 0.0

    @classmethod
    def builtin(cls, client: NIMClient, kind: str, **kwargs: Any) -> Judge:
        """Construct with one of the built-in rubrics by name.

        `kind` is one of ``"correctness" | "faithfulness" | "relevance"``.
        """
        try:
            rubric = BUILTIN_RUBRICS[kind]
        except KeyError as exc:
            raise ValueError(
                f"unknown rubric {kind!r}; available: {sorted(BUILTIN_RUBRICS)}"
            ) from exc
        return cls(client=client, rubric=rubric, **kwargs)

    def grade(
        self,
        *,
        prediction: str,
        question: str | None = None,
        reference: str | None = None,
        context: str | None = None,
    ) -> JudgeResult:
        """Score a single prediction.

        The user message is assembled from whichever of `question`,
        `reference`, `context`, and `prediction` are supplied — matching
        the per-rubric expectations of the project's grader scripts:

        - correctness: question + reference + prediction
        - faithfulness: context + prediction
        - relevance: question + prediction

        Pass whichever fields the rubric needs; extras are ignored cheaply.
        """
        user = self._build_user_message(
            question=question,
            reference=reference,
            context=context,
            prediction=prediction,
        )
        try:
            raw_response = self.client.chat(
                [
                    {"role": "system", "content": self.rubric},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except NIMError as exc:
            raise JudgeError(f"judge call failed: {exc}") from exc
        try:
            text = raw_response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError(
                f"judge response missing content: {raw_response!r}"
            ) from exc
        return self.parse(text)

    @staticmethod
    def parse(raw: str) -> JudgeResult:
        """Extract `{"score": N, "rationale": "..."}` from a judge response.

        Strategy: strip ``` fences, try strict JSON parse on the largest
        ``{...}`` substring, then fall back to regex score extraction.
        Returns ``JudgeResult(score=None, ...)`` when no score field can
        be located.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e > s:
            candidate = text[s : e + 1]
            try:
                obj = json.loads(candidate)
                score = obj.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    return JudgeResult(
                        score=float(score),
                        rationale=str(obj.get("rationale", "")),
                        raw=raw,
                    )
            except json.JSONDecodeError:
                pass
        m = _SCORE_RE_TIGHT.search(text) or _SCORE_RE_LOOSE.search(text)
        if m:
            return JudgeResult(
                score=float(m.group("score")), rationale=text, raw=raw
            )
        return JudgeResult(score=None, rationale=text, raw=raw)

    def _build_user_message(
        self,
        *,
        question: str | None,
        reference: str | None,
        context: str | None,
        prediction: str,
    ) -> str:
        parts: list[str] = []
        if question is not None:
            parts.append(f"Question: {question}")
        if reference is not None:
            parts.append(f"Reference answer: {reference}")
        if context is not None:
            parts.append(f"Context passages:\n\n{context}")
        parts.append(f"Predicted answer: {prediction}")
        parts.append("\nGrade:")
        return "\n".join(parts)


# --- Trajectory ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectoryIter:
    """One agent-loop iteration record, post-eval.

    Mirrors the JSONL schema the `autoresearch-agent-loop` article emits:
    each iteration has integer `iter`, a `proposal: {knob, new_value, ...}`,
    a `decision: "keep" | "revert"`, and a numeric `val_bpb` (or another
    score field — `Trajectory.from_jsonl(score_field="...")` chooses).

    `raw` is the full original record so callers can read niche fields
    (timings, candidate_cfg, etc.) without re-parsing the JSONL.
    """

    iter: int
    knob: str
    value: Any
    decision: str
    score: float
    raw: dict[str, Any]


@dataclass
class Trajectory:
    """Analyzer for the agent-loop trajectory JSONL.

    Schema (one record per JSON line):

    1. **Header** (first line, optional): a meta dict that contains a
       numeric ``baseline_<score_field>`` (e.g. `baseline_val_bpb`) and
       any other run-wide metadata. Detected by the absence of an
       ``"iter"`` key.
    2. **Iteration records** (subsequent lines): dicts with
       ``"iter": int``, ``"stage": "evaluated"`` (records of other
       stages are skipped), ``"proposal": {"knob": str, "new_value": ...}``,
       ``"decision": str``, and the score field (default `val_bpb`).

    Parsing is permissive: malformed lines and records missing required
    keys are dropped silently, since the agent loop also writes
    intermediate stages (`"proposed"`, `"failed"`) that aren't iteration
    outcomes.

    All analysis methods operate on the parsed in-memory list and are
    pure — cheap to call repeatedly with different parameters.
    """

    iters: list[TrajectoryIter]
    baseline: float | None = None
    score_field: str = "val_bpb"
    lower_is_better: bool = True
    header: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        score_field: str = "val_bpb",
        lower_is_better: bool = True,
    ) -> Trajectory:
        """Parse a JSONL trajectory file. See class docstring for the schema."""
        rows: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
        if not rows:
            return cls(
                iters=[],
                baseline=None,
                score_field=score_field,
                lower_is_better=lower_is_better,
            )

        if "iter" in rows[0]:
            header: dict[str, Any] = {}
            iter_rows = rows
        else:
            header = rows[0]
            iter_rows = rows[1:]
        baseline = header.get(f"baseline_{score_field}") if header else None

        parsed: list[TrajectoryIter] = []
        for r in iter_rows:
            stage = r.get("stage")
            if stage not in (None, "evaluated"):
                continue
            try:
                proposal = r.get("proposal") or {}
                parsed.append(
                    TrajectoryIter(
                        iter=int(r["iter"]),
                        knob=str(proposal.get("knob", "")),
                        value=proposal.get("new_value"),
                        decision=str(r.get("decision", "")),
                        score=float(r[score_field]),
                        raw=r,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        return cls(
            iters=parsed,
            baseline=float(baseline) if baseline is not None else None,
            header=header,
            score_field=score_field,
            lower_is_better=lower_is_better,
        )

    # -- Coverage ---------------------------------------------------------

    def knob_coverage(
        self, all_knobs: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Per-knob proposal counts and (optionally) untouched-knob list.

        Pass `all_knobs` (e.g. from your perturbation menu) to also get
        the list of knobs the proposer never touched. Without it, the
        result reports only the knobs that appeared at least once.
        """
        counts = Counter(it.knob for it in self.iters if it.knob)
        out: dict[str, Any] = {
            "knobs_touched": len(counts),
            "knob_count": dict(counts.most_common()),
        }
        if all_knobs:
            untouched = [k for k in all_knobs if k not in counts]
            out["knobs_total"] = len(all_knobs)
            out["knobs_untouched"] = untouched
            out["knobs_touched_pct"] = round(
                100 * len(counts) / len(all_knobs), 1
            )
        return out

    # -- Repeats ----------------------------------------------------------

    def repeat_rate(
        self, *, window: int | None = None
    ) -> float | list[dict[str, Any]]:
        """Fraction of (knob, value) proposals that repeat a prior pair.

        With ``window=None`` returns one float for the whole trajectory.
        With ``window=N`` returns a per-window list of
        ``{first, last, n, repeats, rate}`` records — useful for showing
        the repeat rate climbing as the proposer's history horizon
        forgets older proposals.
        """
        if not self.iters:
            return 0.0 if window is None else []
        seen: set[tuple[str, Any]] = set()
        flags: list[bool] = []
        for it in self.iters:
            pair = (it.knob, _hashable(it.value))
            flags.append(pair in seen)
            seen.add(pair)
        if window is None:
            return round(sum(flags) / len(flags), 3)
        if window <= 0:
            raise ValueError("window must be positive")
        out: list[dict[str, Any]] = []
        for ws in range(0, len(flags), window):
            chunk = flags[ws : ws + window]
            out.append(
                {
                    "first": ws + 1,
                    "last": ws + len(chunk),
                    "n": len(chunk),
                    "repeats": sum(chunk),
                    "rate": round(sum(chunk) / len(chunk), 3),
                }
            )
        return out

    # -- Mode dominance --------------------------------------------------

    def mode_dominance(self, *, top_n: int | None = None) -> list[dict[str, Any]]:
        """Top (knob, value) pairs by proposal count.

        Each entry: ``{"knob": str, "value": Any, "n": int}``. Useful
        for spotting the proposer's collapse mode (e.g. the 5-of-8 keep
        repetition the agent-loop article surfaced).
        """
        pairs = Counter(
            (it.knob, _hashable(it.value)) for it in self.iters if it.knob
        )
        items = (
            pairs.most_common(top_n) if top_n is not None else pairs.most_common()
        )
        return [{"knob": k, "value": v, "n": n} for (k, v), n in items]

    # -- Cumulative best -------------------------------------------------

    def cumulative_best(self, *, baseline: float | None = None) -> list[float]:
        """Best-so-far score after each iteration (length == len(self.iters))."""
        if not self.iters:
            return []
        cur = baseline if baseline is not None else self.baseline
        if cur is None:
            cur = self.iters[0].score
        out: list[float] = []
        for it in self.iters:
            cur = (
                min(cur, it.score) if self.lower_is_better else max(cur, it.score)
            )
            out.append(cur)
        return out

    # -- Convenience -----------------------------------------------------

    def keeps(self) -> list[TrajectoryIter]:
        """Iterations the loop kept. ``decision == "keep"`` filter."""
        return [it for it in self.iters if it.decision == "keep"]

    def best(self) -> TrajectoryIter | None:
        """Iteration with the best score. None on empty trajectory."""
        if not self.iters:
            return None
        return (
            min(self.iters, key=lambda it: it.score)
            if self.lower_is_better
            else max(self.iters, key=lambda it: it.score)
        )


def _hashable(v: Any) -> Any:
    """Best-effort coercion to a hashable for set / dict use.

    Lists become tuples, dicts become tuple-of-sorted-items, anything
    else that's already hashable is returned unchanged. Falls back to
    `str(v)` for exotic types (numpy arrays, etc.).
    """
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(val)) for k, val in v.items()))
    try:
        hash(v)
        return v
    except TypeError:
        return str(v)


# --- AssertionGrader -----------------------------------------------------


ASSERTION_KINDS: tuple[str, ...] = (
    "file_exists",
    "file_not_exists",
    "file_contents_contain",
    "file_contents_match_regex",
    "file_unchanged",
)
"""The five programmatic assertion primitives the grader supports.

Lifted verbatim from `articles/clawgym-on-spark/scripts/grader.py` —
the `clawgym-on-spark` synth corpus uses exactly this set, with each
assertion verifying a post-rollout file-system state. LLM-as-judge
flavored grading lives separately in `fieldkit.eval.Judge`.
"""


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """One assertion's outcome: kind, path, pass/fail, and optional detail.

    `detail` is empty on pass; on failure it records the proximate cause
    (missing path, regex did not match, divergent contents, etc.) so a
    grade dump is debuggable without re-running the rollout.
    """

    kind: str
    path: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class GradeResult:
    """A task's overall grade: per-assertion outcomes plus binary AND.

    `passed` is True iff every assertion passed. `n_passed` / `n_total`
    enable per-assertion-rate metrics across a task batch — the
    `articles/clawgym-on-spark` lift used these for the per-persona +
    per-assertion-kind breakdowns the article reports.
    """

    task_id: str
    passed: bool
    n_passed: int
    n_total: int
    assertions: list[AssertionResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "assertions": [a.to_dict() for a in self.assertions],
        }


@dataclass
class AssertionGrader:
    """Pure-function grader over five file-system assertion primitives.

    Usage::

        from pathlib import Path
        from fieldkit.eval import AssertionGrader

        grader = AssertionGrader()
        result = grader.grade(
            task,                                 # SynthTask-shaped dict OR bare list
            post_state_root=Path("/tmp/sandbox-N"),
        )
        print(result.passed, result.n_passed, result.n_total)

    The grader is intentionally a pure function over the file system —
    no LLM, no fuzzy matching, no scoring. The five supported kinds are
    listed in `ASSERTION_KINDS`; an unknown kind fails the assertion with
    a `"unknown kind: <k>"` detail rather than crashing the grade.

    `task` accepts either:

    - **a SynthTask-shaped dict** — must have ``verifiable_assertions``,
      and may have ``task_id`` and ``workspace_seed.files`` (the latter
      auto-populates ``seed_files`` for ``file_unchanged`` checks);
    - **a bare list of assertion dicts** — each entry has
      ``kind``, ``path``, plus kind-specific keys (``must_contain``,
      ``regex``).

    `seed_files` is the pre-rollout text-content map for ``file_unchanged``
    assertions; without it those assertions report "skipped (no seed
    content)" and count as pass. Pass it explicitly to enforce.
    """

    @staticmethod
    def supported_kinds() -> tuple[str, ...]:
        """The exact tuple from `ASSERTION_KINDS`. Useful for menus / docs."""
        return ASSERTION_KINDS

    def grade(
        self,
        task: dict[str, Any] | Sequence[dict[str, Any]],
        post_state_root: str | Path,
        *,
        seed_files: dict[str, str] | None = None,
        task_id: str = "",
    ) -> GradeResult:
        """Evaluate one task's assertions against a post-rollout directory.

        The grader walks each assertion in declaration order. A binary AND
        of pass/fail across all assertions becomes `GradeResult.passed`.
        Per-assertion failures are surfaced in `GradeResult.assertions`
        with a `detail` string explaining the proximate cause.
        """
        root = Path(post_state_root)
        assertions, derived_seeds, derived_id = self._unpack_task(task)
        if seed_files is None:
            seed_files = derived_seeds
        if not task_id:
            task_id = derived_id

        results: list[AssertionResult] = []
        for a in assertions:
            kind = a.get("kind", "")
            rel = a.get("path", "")
            full = root / rel
            results.append(self._grade_one(kind, rel, full, a, seed_files))

        n_passed = sum(1 for r in results if r.passed)
        return GradeResult(
            task_id=task_id,
            passed=all(r.passed for r in results),
            n_passed=n_passed,
            n_total=len(results),
            assertions=results,
        )

    def _grade_one(
        self,
        kind: str,
        rel: str,
        full: Path,
        a: dict[str, Any],
        seed_files: dict[str, str],
    ) -> AssertionResult:
        if kind == "file_exists":
            ok = full.exists()
            return AssertionResult(kind, rel, ok, "" if ok else "path missing")
        if kind == "file_not_exists":
            ok = not full.exists()
            return AssertionResult(kind, rel, ok, "" if ok else "file still present")
        if kind == "file_contents_contain":
            if not full.is_file():
                return AssertionResult(kind, rel, False, "file missing")
            try:
                body = full.read_text(errors="replace")
            except OSError as exc:
                return AssertionResult(kind, rel, False, f"read error: {exc}")
            missing = [s for s in a.get("must_contain", []) if s not in body]
            ok = not missing
            return AssertionResult(
                kind, rel, ok, "" if ok else f"missing substrings: {missing}"
            )
        if kind == "file_contents_match_regex":
            if not full.is_file():
                return AssertionResult(kind, rel, False, "file missing")
            try:
                body = full.read_text(errors="replace")
                regex = a.get("regex", "")
                ok = re.search(regex, body) is not None
            except (OSError, re.error) as exc:
                return AssertionResult(kind, rel, False, f"error: {exc}")
            return AssertionResult(kind, rel, ok, "" if ok else "regex not matched")
        if kind == "file_unchanged":
            seed = seed_files.get(rel)
            if seed is None:
                return AssertionResult(
                    kind, rel, True, "skipped (no seed content)"
                )
            if not full.is_file():
                return AssertionResult(
                    kind, rel, False, "file missing post-rollout"
                )
            try:
                body = full.read_text(errors="replace")
            except OSError as exc:
                return AssertionResult(kind, rel, False, f"read error: {exc}")
            ok = body == seed
            return AssertionResult(
                kind, rel, ok, "" if ok else "contents diverged from seed"
            )
        return AssertionResult(kind, rel, False, f"unknown kind: {kind}")

    @staticmethod
    def _unpack_task(
        task: dict[str, Any] | Sequence[dict[str, Any]],
    ) -> tuple[Sequence[dict[str, Any]], dict[str, str], str]:
        """Return ``(assertions, seed_files, task_id)`` from either input shape.

        Dict input: pulls `verifiable_assertions`, derives seeds from
        `workspace_seed.files` text entries, takes `task_id`. List input:
        used as-is, no seeds derivable, no task_id.
        """
        if isinstance(task, dict):
            assertions: Sequence[dict[str, Any]] = task.get(
                "verifiable_assertions", []
            )
            seeds: dict[str, str] = {}
            for f in (task.get("workspace_seed") or {}).get("files", []):
                if f.get("kind") == "text":
                    seeds[f.get("path", "")] = f.get("content", "")
            return assertions, seeds, str(task.get("task_id", ""))
        return task, {}, ""
