#!/usr/bin/env python3
# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""V0 preflight gate — score 5 vertical-bench questions on FP-source weights.

Runs *before* the B4 quantize+measure sweep to catch the
chat-vs-continued-pretrain trap (per `feedback_chat_vs_continued_pretrain_trap`
+ `feedback_preflight_bench_before_quant`). Produces an F16 GGUF via
`convert_hf_to_gguf.py` (which IS the FP source representation in the GGUF
ecosystem — no quantization happens here), spins up llama-server on GPU, runs
5 FinanceBench `metrics-generated` questions, scores with
`fieldkit.eval.numeric_match`, exits 0 on ≥ PREFLIGHT_MIN/PREFLIGHT_N or 1 on
fewer correct.

The F16 GGUF this step produces is the same file B4 emits for the `F16`
variant — so V0 is a strict subset of B4 work, not extra overhead. On failure
we abort before the multi-hour `Q4_K_M/Q5_K_M/Q6_K/Q8_0` quantization sweep.

Why GGUF instead of `transformers`: GB10 has unified memory + GPU. Loading
the FP16 source via transformers on CPU is ~3 tok/s (single-question wall
~30s for 256 toks). Loading the F16 GGUF on GPU is ~10 tok/s (single-question
~25s) plus a one-time ~5min convert. For five questions the GGUF path is
already cheaper, and for the next retry (when the GGUF is cached) it's
~30 sec end-to-end vs ~30 minutes on CPU fp32.

Inputs (env, mirrors `g3_build_first_quant.sh` defaults):

    MODELS_DIR      /home/nvidia/data/models
    MODEL_SLUG      basename of MODEL_ID
    QUANTS_DIR      /home/nvidia/data/quants
    LLAMA_CPP_BIN   /home/nvidia/llama.cpp/build/bin
    LLAMA_CPP_CONVERT
                    /home/nvidia/llama.cpp/convert_hf_to_gguf.py
    BASE_MODEL_ARG  HF repo id (for GGUF metadata only)
    FINBENCH_JSONL  /home/nvidia/data/eval-benches/financebench/financebench_merged.jsonl
    FINBENCH_SUBSET metrics-generated
    PREFLIGHT_N     5
    PREFLIGHT_MIN   1
    PREFLIGHT_N_PREDICT
                    256
    PREFLIGHT_NGL   99    (GPU layers; 99 = all)
    PREFLIGHT_CTX   4096

Exit codes:
    0 — pass (≥ PREFLIGHT_MIN correct out of PREFLIGHT_N)
    1 — fail (model + format pairing broken — re-pick)
    2 — preflight could not run (missing weights / bench / llama.cpp)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "fieldkit" / "src"))

from fieldkit.eval import numeric_match  # noqa: E402


# --- FinanceBench loader (evidence-aware) ---------------------------------
# `VerticalBench.from_jsonl` drops the `evidence` field; FinanceBench is an
# open-book benchmark — the right answer can only be derived from the 10-K
# excerpt in `evidence[*].evidence_text`. Inline a thin loader that returns
# (question_with_evidence, expected) tuples so the preflight gate scores
# against the same context the real eval uses.


def _load_finbench_open_book(
    path: Path, subset: str, limit: int
) -> list[tuple[str, str, str]]:
    """Return [(qid, prompt_question, expected_answer), ...].

    `prompt_question` includes the evidence text — the answer to the
    quantitative FinanceBench questions is literally in the cited 10-K row.
    Without it the model is guessing, which masks the chat-vs-base-model
    signal V0 is designed to surface.
    """
    out: list[tuple[str, str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if subset != "all" and row.get("question_type") != subset:
                continue
            q = row.get("question") or ""
            ans = row.get("answer") or row.get("gold_standard") or ""
            if not q or not ans:
                continue
            evidence_chunks: list[str] = []
            for e in row.get("evidence") or []:
                if isinstance(e, dict):
                    txt = e.get("evidence_text") or ""
                    if txt:
                        evidence_chunks.append(str(txt))
                elif isinstance(e, str):
                    evidence_chunks.append(e)
            evidence_text = "\n\n".join(evidence_chunks)
            qid = str(row.get("financebench_id") or f"fb-{len(out)}")
            if evidence_text:
                prompt_q = (
                    f"Context from {row.get('doc_name', 'the filing')}:\n\n"
                    f"{evidence_text}\n\n"
                    f"Question: {q}\n\n"
                    f"Answer with just the numeric value."
                )
            else:
                prompt_q = q
            out.append((qid, prompt_q, str(ans)))
            if len(out) >= limit:
                break
    return out


def _log(msg: str) -> None:
    print(f"[preflight] {msg}", flush=True)


def _die(msg: str, code: int = 2) -> None:
    print(f"[preflight FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _detect_prompt_format(model_dir: Path) -> str:
    """Pick the right chat-template wrapper for the source weights.

    Llama-2-chat models historically ship no `chat_template` in
    tokenizer_config.json — the convention is `<s>[INST] X [/INST]`. The
    README is the most reliable signal that the upstream is Llama-2-chat
    (e.g. AdaptLLM's continued-pretrain-from-Llama-2-chat recipe).
    """
    readme = model_dir / "README.md"
    if readme.exists():
        txt = readme.read_text(errors="ignore").lower()
        if "llama-2-chat" in txt or "llama2-chat" in txt or "[inst]" in txt:
            return "llama2_inst"
    tok_cfg = model_dir / "tokenizer_config.json"
    if tok_cfg.exists():
        try:
            cfg = json.loads(tok_cfg.read_text())
            if cfg.get("chat_template"):
                return "tokenizer_template"
        except Exception:
            pass
    return "raw"


def _format_prompt(question: str, fmt: str) -> str:
    if fmt == "llama2_inst":
        return f"<s>[INST] {question.strip()} [/INST]"
    return question.strip()


def _convert_to_f16_gguf(
    *,
    model_dir: Path,
    out_path: Path,
    convert_script: Path,
    base_model_id: str | None,
) -> None:
    """Run `convert_hf_to_gguf.py --outtype f16 --outfile <out>`."""
    if out_path.exists():
        _log(f"reusing existing F16 GGUF at {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        "f16",
    ]
    _log(f"converting {model_dir.name} → F16 GGUF (this can take ~5 min)")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        _die(f"convert_hf_to_gguf failed (rc={proc.returncode})")
    _log(f"convert OK in {time.perf_counter() - t0:.1f}s → {out_path}")


class LlamaServerSession:
    """Spin up `llama-server` on a free port and tear it down on exit."""

    def __init__(
        self,
        *,
        gguf_path: Path,
        llama_server_bin: Path,
        n_gpu_layers: int = 99,
        ctx_size: int = 4096,
        n_predict: int = 256,
        threads: int = 8,
        startup_timeout_s: float = 180.0,
    ) -> None:
        self.gguf_path = gguf_path
        self.llama_server_bin = llama_server_bin
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.n_predict = n_predict
        self.threads = threads
        self.startup_timeout_s = startup_timeout_s
        self.port = _free_port()
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh = None

    def __enter__(self) -> "LlamaServerSession":
        log_path = Path("/tmp/g3-logs") / f"preflight-server-{self.gguf_path.stem}-{self.port}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(log_path, "wb")
        cmd = [
            str(self.llama_server_bin),
            "-m",
            str(self.gguf_path),
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.n_gpu_layers),
            "-t",
            str(self.threads),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        self._proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server died during startup (rc={self._proc.returncode}); see {log_path}"
                )
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1.0) as resp:
                    if resp.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"llama-server failed to come up within {self.startup_timeout_s}s")

    def __exit__(self, *_exc: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        if self._log_fh is not None:
            self._log_fh.close()

    def complete(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": self.n_predict,
                "stream": False,
                "cache_prompt": False,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"  HTTP error: {type(exc).__name__}: {exc}")
            return ""
        return str(body.get("content") or "").strip()


def main() -> int:
    models_dir = Path(os.environ.get("MODELS_DIR", "/home/nvidia/data/models"))
    quants_dir = Path(os.environ.get("QUANTS_DIR", "/home/nvidia/data/quants"))
    model_slug = os.environ.get("MODEL_SLUG") or _die("MODEL_SLUG env required") or ""
    base_model_arg = os.environ.get("BASE_MODEL_ARG") or os.environ.get("MODEL_ID")
    llama_cpp_bin = Path(os.environ.get("LLAMA_CPP_BIN", "/home/nvidia/llama.cpp/build/bin"))
    llama_convert = Path(
        os.environ.get("LLAMA_CPP_CONVERT", "/home/nvidia/llama.cpp/convert_hf_to_gguf.py")
    )
    finbench_jsonl = Path(
        os.environ.get(
            "FINBENCH_JSONL",
            "/home/nvidia/data/eval-benches/financebench/financebench_merged.jsonl",
        )
    )
    subset = os.environ.get("FINBENCH_SUBSET", "metrics-generated")
    n = int(os.environ.get("PREFLIGHT_N", "5"))
    min_correct = int(os.environ.get("PREFLIGHT_MIN", "1"))
    n_predict = int(os.environ.get("PREFLIGHT_N_PREDICT", "256"))
    n_gpu_layers = int(os.environ.get("PREFLIGHT_NGL", "99"))
    ctx_size = int(os.environ.get("PREFLIGHT_CTX", "4096"))

    model_dir = models_dir / model_slug
    f16_gguf = quants_dir / model_slug / f"model-F16.gguf"
    llama_server_bin = llama_cpp_bin / "llama-server"

    if not (model_dir / "config.json").exists():
        _die(f"model weights not found at {model_dir} (run `g3_build_first_quant.sh download` first)")
    if not finbench_jsonl.exists():
        _die(f"FinanceBench JSONL not found at {finbench_jsonl}")
    if not llama_server_bin.exists():
        _die(f"llama-server not found at {llama_server_bin} (build llama.cpp first)")
    if not llama_convert.exists():
        _die(f"convert_hf_to_gguf.py not found at {llama_convert}")

    _convert_to_f16_gguf(
        model_dir=model_dir,
        out_path=f16_gguf,
        convert_script=llama_convert,
        base_model_id=base_model_arg,
    )

    fmt = _detect_prompt_format(model_dir)
    _log(f"prompt format: {fmt}")
    if fmt == "raw":
        _log("WARN: no chat-format signal found — likely continued-pretrain trap")
        _log("      proceeding anyway; numeric_match will score 0 if outputs aren't formatted")

    items = _load_finbench_open_book(finbench_jsonl, subset=subset, limit=n)
    if not items:
        _die(f"no questions match subset={subset!r} in {finbench_jsonl}")
    _log(f"scoring {len(items)} open-book questions from FinanceBench subset={subset}")

    correct = 0
    with LlamaServerSession(
        gguf_path=f16_gguf,
        llama_server_bin=llama_server_bin,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
        n_predict=n_predict,
    ) as server:
        for i, (qid, prompt_q, expected) in enumerate(items, start=1):
            prompt = _format_prompt(prompt_q, fmt)
            t_q = time.perf_counter()
            predicted = server.complete(prompt)
            elapsed = time.perf_counter() - t_q
            score = numeric_match(predicted, expected, rel_tolerance=0.01)
            correct += int(score)
            _log(
                f"  Q{i}/{len(items)} [{elapsed:.1f}s] qid={qid} "
                f"expected={expected[:80]!r} predicted={predicted[:200]!r} score={score:.0f}"
            )

    _log(f"score: {correct}/{len(items)} (threshold ≥ {min_correct})")
    if correct >= min_correct:
        _log("PASS — proceed with quantize+measure")
        return 0
    _log("FAIL — model + format pairing broken — re-pick")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[preflight FATAL] unhandled: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
