# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `fieldkit.eval` — Bench (offline), Judge (respx-mocked
NIM), Trajectory (fixture JSONL). No live services required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from fieldkit.eval import (
    BUILTIN_RUBRICS,
    REFUSAL_PATTERNS,
    RUBRIC_CORRECTNESS,
    RUBRIC_FAITHFULNESS,
    RUBRIC_RELEVANCE,
    Bench,
    BenchCall,
    Judge,
    JudgeError,
    JudgeResult,
    Trajectory,
    TrajectoryIter,
    is_refusal,
    summarize_metric,
)
from fieldkit.nim import NIMClient


NIM_BASE_URL = "http://nim.test/v1"
NIM_MODEL = "meta/llama-3.1-8b-instruct"


# --- summarize_metric ----------------------------------------------------


class TestSummarizeMetric:
    def test_empty_returns_n_zero(self) -> None:
        assert summarize_metric([]) == {"n": 0}

    def test_all_none_returns_n_zero(self) -> None:
        assert summarize_metric([None, None]) == {"n": 0}

    def test_basic_stats(self) -> None:
        s = summarize_metric([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["n"] == 5
        assert s["mean"] == 3.0
        assert s["median"] == 3.0
        assert s["min"] == 1.0
        assert s["max"] == 5.0

    def test_drops_none_entries(self) -> None:
        s = summarize_metric([10.0, None, 20.0, None])
        assert s["n"] == 2
        assert s["min"] == 10.0
        assert s["max"] == 20.0


# --- Bench ---------------------------------------------------------------


def _identity_with_metrics(x: int) -> dict[str, Any]:
    return {"value": x, "tokens": x * 10}


def _nested_metrics(x: int) -> dict[str, Any]:
    return {
        "answer": f"answer-{x}",
        "timings_ms": {"embed": x * 1.0, "retrieve": x * 2.0, "generate": x * 3.0},
    }


class TestBenchRun:
    def test_collects_calls(self) -> None:
        b = Bench("identity")
        b.run(_identity_with_metrics, [1, 2, 3])
        assert len(b.calls) == 3
        assert all(c.success for c in b.calls)
        assert [c.input for c in b.calls] == [1, 2, 3]
        assert b.calls[0].output == {"value": 1, "tokens": 10}

    def test_extracts_top_level_metrics(self) -> None:
        b = Bench("identity", metrics=["tokens"])
        b.run(_identity_with_metrics, [1, 2, 3])
        assert [c.metrics["tokens"] for c in b.calls] == [10.0, 20.0, 30.0]

    def test_extracts_nested_metrics_via_metrics_key(self) -> None:
        b = Bench("rag", metrics=["embed", "retrieve"], metrics_key="timings_ms")
        b.run(_nested_metrics, [1, 2, 3])
        assert b.calls[0].metrics == {"embed": 1.0, "retrieve": 2.0}
        assert b.calls[2].metrics == {"embed": 3.0, "retrieve": 6.0}

    def test_records_latency_ms(self) -> None:
        def slow(x: int) -> int:
            time.sleep(0.01)
            return x

        b = Bench("slow")
        b.run(slow, [1])
        assert b.calls[0].latency_ms >= 9.0  # 10ms sleep, allow noise

    def test_run_returns_self(self) -> None:
        b = Bench("chain")
        result = b.run(_identity_with_metrics, [1])
        assert result is b

    def test_tag_fn_attaches_metadata(self) -> None:
        b = Bench("tagged")
        b.run(_identity_with_metrics, [1, 2], tag_fn=lambda x: {"kind": "even" if x % 2 == 0 else "odd"})
        assert b.calls[0].tags == {"kind": "odd"}
        assert b.calls[1].tags == {"kind": "even"}


class TestBenchOnError:
    def test_default_records_failure(self) -> None:
        def boom(x: int) -> int:
            if x == 2:
                raise ValueError("nope")
            return x

        b = Bench("boom")
        b.run(boom, [1, 2, 3])
        assert [c.success for c in b.calls] == [True, False, True]
        assert b.calls[1].error is not None
        assert "ValueError" in b.calls[1].error
        assert "nope" in b.calls[1].error

    def test_raise_aborts_sweep(self) -> None:
        def boom(x: int) -> int:
            if x == 2:
                raise ValueError("nope")
            return x

        b = Bench("boom")
        with pytest.raises(ValueError, match="nope"):
            b.run(boom, [1, 2, 3], on_error="raise")
        # Got call 1 in before raising.
        assert len(b.calls) == 1
        assert b.calls[0].success

    def test_invalid_on_error_value(self) -> None:
        b = Bench("x")
        with pytest.raises(ValueError, match="on_error"):
            b.run(lambda x: x, [1], on_error="huh")


class TestBenchRecord:
    def test_imperative_record(self) -> None:
        b = Bench("manual", metrics=["embed", "retrieve"])
        b.record(input="q1", latency_ms=42.0, embed=10.0, retrieve=32.0)
        assert b.calls[0].latency_ms == 42.0
        assert b.calls[0].metrics == {"embed": 10.0, "retrieve": 32.0}


class TestBenchSummary:
    def test_empty_summary(self) -> None:
        b = Bench("empty")
        assert b.summary() == {"name": "empty", "n": 0}

    def test_aggregates_latency_and_metrics(self) -> None:
        b = Bench("agg", metrics=["tokens"])
        for i in [1, 2, 3]:
            b.record(latency_ms=float(i * 100), tokens=float(i * 10))
        s = b.summary()
        assert s["n"] == 3
        assert s["n_success"] == 3
        assert s["n_failure"] == 0
        assert s["latency_ms"]["mean"] == 200.0
        assert s["latency_ms"]["min"] == 100.0
        assert s["latency_ms"]["max"] == 300.0
        assert s["tokens"]["mean"] == 20.0
        assert s["tokens"]["median"] == 20.0

    def test_failures_counted(self) -> None:
        b = Bench("agg")
        b.record(latency_ms=10.0)
        b.record(latency_ms=20.0, success=False, error="boom")
        s = b.summary()
        assert s["n"] == 2
        assert s["n_success"] == 1
        assert s["n_failure"] == 1
        # Failure latency is excluded from successful aggregate.
        assert s["latency_ms"]["n"] == 1
        assert s["latency_ms"]["mean"] == 10.0


class TestBenchReport:
    def test_renders_markdown_table(self) -> None:
        b = Bench("doc", metrics=["tokens"])
        b.record(latency_ms=10.0, tokens=5.0)
        b.record(latency_ms=20.0, tokens=15.0)
        text = b.report()
        assert "### Bench: doc (n=2)" in text
        assert "| metric | mean | median | min | max |" in text
        assert "| latency_ms |" in text
        assert "| tokens |" in text

    def test_dashes_when_metric_missing(self) -> None:
        b = Bench("doc", metrics=["tokens"])
        b.record(latency_ms=10.0)  # no tokens metric provided
        text = b.report()
        assert "| tokens | — | — | — | — |" in text


class TestBenchDump:
    def test_dump_round_trip(self, tmp_path: Path) -> None:
        b = Bench("dumper", metrics=["tokens"])
        b.record(input="q", output={"answer": "a"}, latency_ms=12.0, tokens=7.0)
        path = b.dump(tmp_path / "bench.json")
        loaded = json.loads(path.read_text())
        assert loaded["summary"]["name"] == "dumper"
        assert loaded["summary"]["n"] == 1
        assert "calls" in loaded
        assert len(loaded["calls"]) == 1
        # Output dropped by default.
        assert "output" not in loaded["calls"][0]
        assert loaded["calls"][0]["latency_ms"] == 12.0
        assert loaded["calls"][0]["metrics"] == {"tokens": 7.0}

    def test_dump_can_include_outputs(self, tmp_path: Path) -> None:
        b = Bench("dumper")
        b.record(input="q", output={"answer": "long-answer"}, latency_ms=1.0)
        path = b.dump(tmp_path / "bench.json", include_outputs=True)
        loaded = json.loads(path.read_text())
        assert loaded["calls"][0]["output"] == {"answer": "long-answer"}


class TestBenchContextManager:
    def test_wall_seconds_recorded(self) -> None:
        with Bench("walled") as b:
            time.sleep(0.005)
        assert b.wall_seconds >= 0.004
        assert "wall_seconds" not in b.summary()  # n==0 short-circuits
        b.record(latency_ms=1.0)
        s = b.summary()
        assert s["wall_seconds"] >= 0.004


# --- is_refusal ----------------------------------------------------------


class TestIsRefusal:
    def test_empty_is_refusal(self) -> None:
        assert is_refusal("")
        assert is_refusal(None)

    @pytest.mark.parametrize(
        "text",
        [
            "I do not know the answer.",
            "I don't have enough information.",
            "I cannot answer that.",
            "I am not able to provide that.",
            "The provided context does not contain the answer.",
            "Not specified in the context.",
            "No information available.",
            "Unclear from the passages.",
            "Insufficient context for that question.",
            "Cannot be determined from the source.",
        ],
    )
    def test_known_refusal_patterns(self, text: str) -> None:
        assert is_refusal(text), f"should be refusal: {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "George W. Bush won the 2004 election.",
            "The Spark has 128 GB of unified memory.",
            "Phelps won 8 medals at the 2008 Olympics.",
        ],
    )
    def test_real_answers_not_flagged(self, text: str) -> None:
        assert not is_refusal(text), f"should NOT be refusal: {text!r}"


# --- Judge ---------------------------------------------------------------


def _judge_response(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-judge",
        "object": "chat.completion",
        "model": NIM_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    }


@pytest.fixture
def judge_client() -> NIMClient:
    c = NIMClient(base_url=NIM_BASE_URL, model=NIM_MODEL, max_retries=0, timeout=2.0)
    yield c
    c.close()


class TestJudgeBuiltins:
    def test_builtin_rubrics_present(self) -> None:
        assert set(BUILTIN_RUBRICS) == {"correctness", "faithfulness", "relevance"}
        assert BUILTIN_RUBRICS["correctness"] is RUBRIC_CORRECTNESS
        assert BUILTIN_RUBRICS["faithfulness"] is RUBRIC_FAITHFULNESS
        assert BUILTIN_RUBRICS["relevance"] is RUBRIC_RELEVANCE

    def test_builtin_factory(self, judge_client: NIMClient) -> None:
        j = Judge.builtin(judge_client, "correctness")
        assert j.rubric == RUBRIC_CORRECTNESS

    def test_builtin_factory_unknown_kind(self, judge_client: NIMClient) -> None:
        with pytest.raises(ValueError, match="unknown rubric"):
            Judge.builtin(judge_client, "made-up")


class TestJudgeParse:
    def test_strict_json_with_int_score(self) -> None:
        r = Judge.parse('{"score": 4, "rationale": "almost right"}')
        assert r.score == 4.0
        assert r.rationale == "almost right"

    def test_strict_json_with_float_score(self) -> None:
        r = Judge.parse('{"score": 0.5, "rationale": "partial"}')
        assert r.score == 0.5
        assert r.rationale == "partial"

    def test_strips_json_fences(self) -> None:
        text = '```json\n{"score": 5, "rationale": "exact"}\n```'
        r = Judge.parse(text)
        assert r.score == 5.0
        assert r.rationale == "exact"

    def test_strips_bare_fences(self) -> None:
        text = '```\n{"score": 3, "rationale": "ok"}\n```'
        r = Judge.parse(text)
        assert r.score == 3.0

    def test_regex_fallback_when_json_invalid(self) -> None:
        # Trailing prose, no closing brace handling — regex finds the score.
        r = Judge.parse('here you go: "score": 2, then more text')
        assert r.score == 2.0

    def test_returns_none_when_unparseable(self) -> None:
        r = Judge.parse("I cannot give a score, sorry.")
        assert r.score is None
        assert "cannot give" in r.rationale

    def test_largest_brace_substring_wins(self) -> None:
        # Two JSON-ish braces; largest substring carries the actual score.
        r = Judge.parse('preface {garbage} ... {"score": 4, "rationale": "fine"}')
        assert r.score == 4.0


class TestJudgeGrade:
    def test_grade_correctness_round_trip(
        self, judge_client: NIMClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{NIM_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_judge_response('{"score": 4, "rationale": "close enough"}')
            )
        )
        j = Judge.builtin(judge_client, "correctness")
        result = j.grade(
            question="Q?", reference="R", prediction="P"
        )
        assert isinstance(result, JudgeResult)
        assert result.score == 4.0
        assert result.rationale == "close enough"
        # Verify the rubric arrived as the system message.
        body = json.loads(route.calls[0].request.content)
        assert body["messages"][0]["role"] == "system"
        assert "0-5 scale" in body["messages"][0]["content"]
        # User message includes question + reference + prediction.
        user = body["messages"][1]["content"]
        assert "Question: Q?" in user
        assert "Reference answer: R" in user
        assert "Predicted answer: P" in user

    def test_grade_faithfulness_uses_context(
        self, judge_client: NIMClient, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(f"{NIM_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_judge_response('{"score": 1.0, "rationale": "all supported"}')
            )
        )
        j = Judge.builtin(judge_client, "faithfulness")
        r = j.grade(prediction="A", context="passage about A")
        assert r.score == 1.0
        body = json.loads(route.calls[0].request.content)
        assert "Context passages" in body["messages"][1]["content"]
        assert "passage about A" in body["messages"][1]["content"]

    def test_grade_handles_unparseable_judge_output(
        self, judge_client: NIMClient, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(f"{NIM_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_judge_response("this judge forgot the JSON envelope")
            )
        )
        j = Judge.builtin(judge_client, "correctness")
        r = j.grade(prediction="x", reference="y", question="z")
        assert r.score is None

    def test_grade_wraps_nim_errors(
        self, judge_client: NIMClient, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(f"{NIM_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(400, text="bad request")
        )
        j = Judge.builtin(judge_client, "correctness")
        with pytest.raises(JudgeError, match="judge call failed"):
            j.grade(prediction="x", reference="y", question="z")


class TestJudgeRefusalPatternsExposed:
    def test_refusal_patterns_compiled(self) -> None:
        # Sanity: tuple of compiled regexes, all case-insensitive.
        assert len(REFUSAL_PATTERNS) > 0
        for p in REFUSAL_PATTERNS:
            assert p.flags & 2  # re.IGNORECASE


# --- Trajectory ----------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _baseline_header() -> dict[str, Any]:
    return {
        "_meta": "trajectory log",
        "baseline_val_bpb": 10.95,
        "baseline_cfg": {"n_layer": 24},
    }


def _iter_record(
    iter: int, knob: str, value: Any, decision: str, val_bpb: float, **extra: Any
) -> dict[str, Any]:
    rec = {
        "iter": iter,
        "stage": "evaluated",
        "proposal": {"knob": knob, "new_value": value, "reason": "test"},
        "decision": decision,
        "val_bpb": val_bpb,
    }
    rec.update(extra)
    return rec


@pytest.fixture
def trajectory_path(tmp_path: Path) -> Path:
    rows = [
        _baseline_header(),
        _iter_record(1, "lr", 1e-3, "keep", 10.90),
        _iter_record(2, "lr", 1e-3, "revert", 10.92),  # repeat
        _iter_record(3, "n_head", 8, "keep", 10.85),
        _iter_record(4, "n_head", 8, "revert", 10.88),  # repeat
        _iter_record(5, "d_model", 512, "keep", 10.80),
    ]
    return _write_jsonl(tmp_path / "traj.jsonl", rows)


class TestTrajectoryParse:
    def test_basic_parse(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        assert len(t.iters) == 5
        assert t.baseline == 10.95
        assert t.header["_meta"] == "trajectory log"
        assert t.iters[0].knob == "lr"
        assert t.iters[0].decision == "keep"

    def test_no_header_still_parses(self, tmp_path: Path) -> None:
        rows = [
            _iter_record(1, "lr", 1e-3, "keep", 10.90),
            _iter_record(2, "n_head", 8, "revert", 10.92),
        ]
        p = _write_jsonl(tmp_path / "traj.jsonl", rows)
        t = Trajectory.from_jsonl(p)
        assert t.baseline is None
        assert len(t.iters) == 2

    def test_skips_non_evaluated_stages(self, tmp_path: Path) -> None:
        rows = [
            _baseline_header(),
            _iter_record(1, "lr", 1e-3, "keep", 10.90),
            {"iter": 2, "stage": "proposed", "proposal": {"knob": "lr", "new_value": 2e-3}},
            {"iter": 3, "stage": "failed", "proposal": {"knob": "lr", "new_value": 3e-3}},
            _iter_record(4, "n_head", 8, "keep", 10.85),
        ]
        p = _write_jsonl(tmp_path / "traj.jsonl", rows)
        t = Trajectory.from_jsonl(p)
        assert [it.iter for it in t.iters] == [1, 4]

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.jsonl"
        path.write_text(
            json.dumps(_baseline_header()) + "\n"
            "this is not json\n"
            + json.dumps(_iter_record(1, "lr", 1e-3, "keep", 10.90)) + "\n"
            "\n"  # blank
            + json.dumps({"iter": 2, "stage": "evaluated", "proposal": {"knob": "x"}, "decision": "k"}) + "\n"  # missing val_bpb
            + json.dumps(_iter_record(3, "n_head", 8, "keep", 10.85)) + "\n"
        )
        t = Trajectory.from_jsonl(path)
        assert [it.iter for it in t.iters] == [1, 3]

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        t = Trajectory.from_jsonl(p)
        assert t.iters == []
        assert t.baseline is None

    def test_alternate_score_field(self, tmp_path: Path) -> None:
        rows = [
            {"baseline_loss": 5.0},
            {"iter": 1, "stage": "evaluated", "proposal": {"knob": "x", "new_value": 1}, "decision": "keep", "loss": 4.5},
            {"iter": 2, "stage": "evaluated", "proposal": {"knob": "y", "new_value": 2}, "decision": "revert", "loss": 4.8},
        ]
        p = _write_jsonl(tmp_path / "traj.jsonl", rows)
        t = Trajectory.from_jsonl(p, score_field="loss")
        assert t.baseline == 5.0
        assert t.iters[0].score == 4.5
        assert t.iters[1].score == 4.8


class TestTrajectoryAnalysis:
    def test_knob_coverage_basic(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        cov = t.knob_coverage()
        assert cov["knobs_touched"] == 3
        assert cov["knob_count"] == {"lr": 2, "n_head": 2, "d_model": 1}
        # No `all_knobs` → no untouched / pct.
        assert "knobs_untouched" not in cov

    def test_knob_coverage_with_universe(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        cov = t.knob_coverage(["lr", "n_head", "d_model", "weight_decay", "beta1"])
        assert cov["knobs_total"] == 5
        assert cov["knobs_untouched"] == ["weight_decay", "beta1"]
        assert cov["knobs_touched_pct"] == 60.0

    def test_repeat_rate_total(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        # Iter 2 (lr=1e-3) and iter 4 (n_head=8) repeat → 2/5 = 0.4
        assert t.repeat_rate() == 0.4

    def test_repeat_rate_windowed(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        windows = t.repeat_rate(window=2)
        assert isinstance(windows, list)
        assert windows[0] == {"first": 1, "last": 2, "n": 2, "repeats": 1, "rate": 0.5}
        assert windows[1] == {"first": 3, "last": 4, "n": 2, "repeats": 1, "rate": 0.5}
        assert windows[2] == {"first": 5, "last": 5, "n": 1, "repeats": 0, "rate": 0.0}

    def test_repeat_rate_rejects_non_positive_window(
        self, trajectory_path: Path
    ) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        with pytest.raises(ValueError, match="window"):
            t.repeat_rate(window=0)

    def test_mode_dominance(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        modes = t.mode_dominance(top_n=2)
        assert len(modes) == 2
        # lr=1e-3 and n_head=8 each have count 2
        assert {(m["knob"], m["n"]) for m in modes} == {("lr", 2), ("n_head", 2)}

    def test_mode_dominance_full(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        modes = t.mode_dominance()
        assert len(modes) == 3  # 3 distinct (knob, value) pairs

    def test_cumulative_best(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        # baseline 10.95 → 10.90, 10.90, 10.85, 10.85, 10.80
        assert t.cumulative_best() == [10.90, 10.90, 10.85, 10.85, 10.80]

    def test_cumulative_best_with_explicit_baseline(
        self, trajectory_path: Path
    ) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        out = t.cumulative_best(baseline=10.84)
        # Iter 1 doesn't beat 10.84 (10.90 > 10.84), iter 3 doesn't (10.85 > 10.84),
        # but iter 5 (10.80) does.
        assert out[0] == 10.84
        assert out[2] == 10.84
        assert out[4] == 10.80

    def test_higher_is_better(self, tmp_path: Path) -> None:
        rows = [
            {"baseline_score": 0.5},
            {"iter": 1, "stage": "evaluated", "proposal": {"knob": "x", "new_value": 1}, "decision": "keep", "score": 0.6},
            {"iter": 2, "stage": "evaluated", "proposal": {"knob": "y", "new_value": 2}, "decision": "revert", "score": 0.55},
            {"iter": 3, "stage": "evaluated", "proposal": {"knob": "z", "new_value": 3}, "decision": "keep", "score": 0.7},
        ]
        p = _write_jsonl(tmp_path / "traj.jsonl", rows)
        t = Trajectory.from_jsonl(p, score_field="score", lower_is_better=False)
        assert t.cumulative_best() == [0.6, 0.6, 0.7]
        assert t.best().score == 0.7

    def test_keeps_filter(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        keeps = t.keeps()
        assert [it.iter for it in keeps] == [1, 3, 5]

    def test_best_iteration(self, trajectory_path: Path) -> None:
        t = Trajectory.from_jsonl(trajectory_path)
        b = t.best()
        assert b is not None
        assert b.iter == 5
        assert b.score == 10.80

    def test_best_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        t = Trajectory.from_jsonl(p)
        assert t.best() is None

    def test_unhashable_value_does_not_crash(self, tmp_path: Path) -> None:
        # Some knobs might propose dict / list values.
        rows = [
            _baseline_header(),
            _iter_record(1, "shape", [1, 2, 3], "keep", 10.90),
            _iter_record(2, "shape", [1, 2, 3], "revert", 10.92),  # same list → repeat
            _iter_record(3, "shape", [1, 2, 4], "keep", 10.88),
        ]
        p = _write_jsonl(tmp_path / "traj.jsonl", rows)
        t = Trajectory.from_jsonl(p)
        assert t.repeat_rate() == round(1 / 3, 3)
        modes = t.mode_dominance()
        # Two distinct (knob, value) tuples after hashing
        assert len(modes) == 2
