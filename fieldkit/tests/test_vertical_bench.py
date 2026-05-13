# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `fieldkit.eval.VerticalBench` — the Spark-overlay vertical
scorer added in v0.4.x for Orionfold vertical-curator quants. Offline only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fieldkit.eval import (
    Bench,
    VerticalBench,
    VerticalQA,
    contains,
    exact_match,
    numeric_match,
)


# --- Scorers -----------------------------------------------------------------


class TestExactMatch:
    def test_identical_returns_1(self) -> None:
        assert exact_match("yes", "yes") == 1.0

    def test_case_insensitive(self) -> None:
        assert exact_match("YES", "yes") == 1.0

    def test_whitespace_insensitive(self) -> None:
        assert exact_match("  yes\n", "yes") == 1.0

    def test_mismatch_returns_0(self) -> None:
        assert exact_match("no", "yes") == 0.0

    def test_substring_does_not_match(self) -> None:
        assert exact_match("yes it does", "yes") == 0.0


class TestContains:
    def test_substring_returns_1(self) -> None:
        assert contains("the answer is yes", "yes") == 1.0

    def test_case_insensitive(self) -> None:
        assert contains("THE ANSWER IS YES", "yes") == 1.0

    def test_missing_returns_0(self) -> None:
        assert contains("the answer is no", "yes") == 0.0

    def test_empty_expected_returns_0(self) -> None:
        assert contains("anything", "") == 0.0


class TestNumericMatch:
    def test_exact_number(self) -> None:
        assert numeric_match("the answer is 1234.5", "1234.5") == 1.0

    def test_within_default_tolerance(self) -> None:
        # 1% tolerance — 1234.5 ± 12.345
        assert numeric_match("1240", "1234.5") == 1.0

    def test_outside_default_tolerance(self) -> None:
        assert numeric_match("1300", "1234.5") == 0.0

    def test_comma_stripping(self) -> None:
        assert numeric_match("Revenue was $1,234,567", "1234567") == 1.0

    def test_first_number_wins(self) -> None:
        # First number in the predicted string wins — here "100", not "200".
        assert numeric_match("Revenue was 100, costs were 200", "100") == 1.0

    def test_no_number_in_predicted_returns_0(self) -> None:
        assert numeric_match("the data does not specify", "42") == 0.0

    def test_no_number_in_expected_returns_0(self) -> None:
        assert numeric_match("42", "no answer") == 0.0

    def test_custom_tolerance(self) -> None:
        # 10% tolerance — 100 ± 10
        assert numeric_match("109", "100", rel_tolerance=0.1) == 1.0
        assert numeric_match("111", "100", rel_tolerance=0.1) == 0.0

    def test_negative_numbers(self) -> None:
        assert numeric_match("the loss was -50.0M", "-50") == 1.0

    def test_zero_reference_uses_absolute_tolerance(self) -> None:
        # When expected is exactly 0, we treat tolerance as absolute (|pn| <= rel_tol).
        assert numeric_match("0.005", "0", rel_tolerance=0.01) == 1.0
        assert numeric_match("0.5", "0", rel_tolerance=0.01) == 0.0


# --- JSONL loader ------------------------------------------------------------


def _write_jsonl(tmp: Path, name: str, rows: list[dict]) -> Path:
    p = tmp / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


class TestFromJsonl:
    def test_financebench_autodetect(self, tmp_path: Path) -> None:
        rows = [
            {
                "financebench_id": "fb-001",
                "question": "What was Pepsi's revenue in FY2022?",
                "gold_standard": "79.47 billion",
                "answer": "Per the 10-K, revenue was approximately $79.47 billion.",
                "company": "Pepsi",
                "doc_period": "FY2022",
                "doc_type": "10-K",
                "question_type": "numerical",
            }
        ]
        p = _write_jsonl(tmp_path, "fb.jsonl", rows)
        vb = VerticalBench.from_jsonl(p)
        assert len(vb.questions) == 1
        q = vb.questions[0]
        assert q.qid == "fb-001"
        assert q.expected == "79.47 billion"
        assert q.tags["company"] == "Pepsi"
        assert q.tags["doc_period"] == "FY2022"
        # numeric_match is the default for financebench
        assert vb.scorer.__name__ == "numeric_match"

    def test_legalbench_autodetect(self, tmp_path: Path) -> None:
        rows = [
            {
                "id": "lb-001",
                "text": "Is this clause a non-compete?",
                "answer": "yes",
                "task": "contract_nli",
            }
        ]
        p = _write_jsonl(tmp_path, "lb.jsonl", rows)
        vb = VerticalBench.from_jsonl(p)
        assert len(vb.questions) == 1
        q = vb.questions[0]
        assert q.qid == "lb-001"
        assert q.question == "Is this clause a non-compete?"
        assert q.expected == "yes"
        assert q.tags["task"] == "contract_nli"
        # exact_match is the default for legalbench
        assert vb.scorer.__name__ == "exact_match"

    def test_generic_fallback(self, tmp_path: Path) -> None:
        rows = [
            {"question": "2+2?", "answer": "4"},
            {"prompt": "3+3?", "expected": "6"},
        ]
        p = _write_jsonl(tmp_path, "generic.jsonl", rows)
        vb = VerticalBench.from_jsonl(p)
        assert len(vb.questions) == 2
        assert vb.questions[0].question == "2+2?"
        assert vb.questions[0].expected == "4"
        assert vb.questions[1].question == "3+3?"
        assert vb.questions[1].expected == "6"

    def test_limit_caps_rows(self, tmp_path: Path) -> None:
        rows = [{"question": f"q{i}", "answer": str(i)} for i in range(10)]
        p = _write_jsonl(tmp_path, "many.jsonl", rows)
        vb = VerticalBench.from_jsonl(p, limit=3)
        assert len(vb.questions) == 3

    def test_missing_required_fields_dropped(self, tmp_path: Path) -> None:
        rows = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2"},  # missing answer
            {"answer": "a3"},  # missing question
            {"question": "q4", "answer": "a4"},
        ]
        p = _write_jsonl(tmp_path, "partial.jsonl", rows)
        vb = VerticalBench.from_jsonl(p)
        assert len(vb.questions) == 2
        assert {q.expected for q in vb.questions} == {"a1", "a4"}

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.jsonl"
        p.write_text(
            '{"question": "good", "answer": "yes"}\n'
            "not json at all\n"
            '{"question": "also good", "answer": "no"}\n'
        )
        vb = VerticalBench.from_jsonl(p)
        assert len(vb.questions) == 2

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "blanks.jsonl"
        p.write_text(
            '\n\n{"question": "q", "answer": "a"}\n\n'
        )
        vb = VerticalBench.from_jsonl(p)
        assert len(vb.questions) == 1

    def test_format_override(self, tmp_path: Path) -> None:
        # A row that looks like generic (no financebench_id) but caller forces financebench.
        rows = [{"question": "q", "gold_standard": "g"}]
        p = _write_jsonl(tmp_path, "force.jsonl", rows)
        vb = VerticalBench.from_jsonl(p, format="financebench")
        assert len(vb.questions) == 1
        assert vb.questions[0].expected == "g"

    def test_custom_scorer_override(self, tmp_path: Path) -> None:
        rows = [{"question": "q", "answer": "a"}]
        p = _write_jsonl(tmp_path, "custom.jsonl", rows)
        vb = VerticalBench.from_jsonl(p, scorer=contains)
        assert vb.scorer is contains

    def test_name_defaults_to_stem(self, tmp_path: Path) -> None:
        rows = [{"question": "q", "answer": "a"}]
        p = _write_jsonl(tmp_path, "financebench-mini.jsonl", rows)
        vb = VerticalBench.from_jsonl(p)
        assert vb.name == "financebench-mini"


# --- VerticalBench.run -------------------------------------------------------


class TestVerticalBenchRun:
    def _make(self, scorer=exact_match) -> VerticalBench:
        return VerticalBench(
            name="toy",
            questions=[
                VerticalQA(qid="q1", question="say yes", expected="yes", tags={"section": "a"}),
                VerticalQA(qid="q2", question="say no", expected="no", tags={"section": "b"}),
            ],
            scorer=scorer,
        )

    def test_perfect_model_scores_1(self) -> None:
        vb = self._make()
        # model_fn echoes the literal expected — should score 1.0 each.
        bench = vb.run(lambda q: "yes" if "yes" in q else "no")
        s = bench.summary()
        assert s["n"] == 2
        assert s["n_success"] == 2
        assert s["accuracy"]["mean"] == 1.0

    def test_wrong_model_scores_0(self) -> None:
        vb = self._make()
        bench = vb.run(lambda q: "wrong")
        s = bench.summary()
        assert s["accuracy"]["mean"] == 0.0

    def test_refusal_tracked(self) -> None:
        vb = self._make()
        # is_refusal hits on "I don't know"
        bench = vb.run(lambda q: "I don't know")
        s = bench.summary()
        assert s["refusal"]["mean"] == 1.0
        # And refusals don't count as correct.
        assert s["accuracy"]["mean"] == 0.0

    def test_tags_propagate(self) -> None:
        vb = self._make()
        bench = vb.run(lambda q: "yes", extra_tags={"variant": "Q4_K_M"})
        # Each BenchCall.tags carries qid + question tags + extra_tags
        for call in bench.calls:
            assert "qid" in call.tags
            assert "variant" in call.tags
            assert call.tags["variant"] == "Q4_K_M"
        sections = [c.tags.get("section") for c in bench.calls]
        assert sorted(sections) == ["a", "b"]

    def test_limit_caps_runs(self) -> None:
        vb = self._make()
        bench = vb.run(lambda q: "yes", limit=1)
        assert bench.summary()["n"] == 1

    def test_returns_bench_instance(self) -> None:
        vb = self._make()
        bench = vb.run(lambda q: "yes")
        assert isinstance(bench, Bench)

    def test_scorer_with_kwargs(self) -> None:
        # numeric_match accepts rel_tolerance — scorer_kwargs should flow through.
        vb = VerticalBench(
            name="num",
            questions=[VerticalQA(qid="q1", question="?", expected="100")],
            scorer=numeric_match,
            scorer_kwargs={"rel_tolerance": 0.5},
        )
        # 140 is within ±50 of 100
        bench = vb.run(lambda q: "140")
        assert bench.summary()["accuracy"]["mean"] == 1.0


# --- VerticalBench.summary (pre-run, lineage-friendly) -----------------------


class TestVerticalBenchSummary:
    def test_summary_shape(self) -> None:
        vb = VerticalBench(
            name="toy",
            questions=[
                VerticalQA(qid="q1", question="q", expected="a", tags={"company": "X", "year": 2024}),
                VerticalQA(qid="q2", question="q", expected="a", tags={"company": "Y"}),
            ],
            scorer=numeric_match,
        )
        s = vb.summary()
        assert s["name"] == "toy"
        assert s["n"] == 2
        assert s["scorer"] == "numeric_match"
        assert s["tag_keys"] == ["company", "year"]


# --- End-to-end --------------------------------------------------------------


class TestEndToEnd:
    def test_financebench_jsonl_to_bench_summary(self, tmp_path: Path) -> None:
        # Two FinanceBench-shaped rows, model gets one right and one wrong.
        rows = [
            {
                "financebench_id": "fb-correct",
                "question": "What was revenue?",
                "gold_standard": "100",
                "company": "ACME",
                "question_type": "numerical",
            },
            {
                "financebench_id": "fb-wrong",
                "question": "What was profit?",
                "gold_standard": "50",
                "company": "ACME",
                "question_type": "numerical",
            },
        ]
        p = _write_jsonl(tmp_path, "fb.jsonl", rows)
        vb = VerticalBench.from_jsonl(p)

        def model_fn(q: str) -> str:
            # Right for the revenue question (100), wrong for the profit question (says 999).
            if "revenue" in q.lower():
                return "It was 100 million."
            return "Profit was 999 million."

        bench = vb.run(model_fn)
        s = bench.summary()
        assert s["n"] == 2
        # Exactly 1 of 2 correct → mean = 0.5
        assert s["accuracy"]["mean"] == 0.5
        # No refusals
        assert s["refusal"]["mean"] == 0.0

    def test_per_row_metric_capture(self, tmp_path: Path) -> None:
        # Confirm BenchCall.metrics carries accuracy + refusal per-row,
        # not just the aggregate.
        rows = [{"question": "say yes", "answer": "yes"}]
        p = _write_jsonl(tmp_path, "single.jsonl", rows)
        vb = VerticalBench.from_jsonl(p)
        bench = vb.run(lambda q: "yes")
        assert len(bench.calls) == 1
        assert bench.calls[0].metrics["accuracy"] == 1.0
        assert bench.calls[0].metrics["refusal"] == 0.0
        assert bench.calls[0].success is True
