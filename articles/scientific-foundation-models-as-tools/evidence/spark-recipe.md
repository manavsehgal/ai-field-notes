# Proposed Spark recipe

1. **Pick a concrete domain** for the first walkthrough — weather forecasting via Pangu-Weather (~3B) is the cleanest demo because input/output is tensor-on-disk rather than tokens.
2. **Serve the planner LLM via NIM**: `llama-3.1-8b-instruct` container exposes the OpenAI-compatible chat endpoint already documented in `nim-first-inference-dgx-spark`. `NIM_GPU_MEM_FRACTION=0.4` to reserve room for the specialist.
3. **Serve the specialist FM via Triton** with a Python backend wrapping the model's native inference (`trtllm-and-triton-on-spark` shows the pattern). Expose it as `predict_weather(state_tensor) -> forecast_tensor`.
4. **Build the EywaAgent loop inside NemoClaw** as a custom skill: planner receives a question, decides whether to call the specialist, marshals the structured inputs, gets back a tensor, and renders a natural-language summary. NemoClaw's tool-call protocol (verified in `nemoclaw-vs-openclaw-dgx-spark` and `autoresearch-agent-loop`) handles the routing.
5. **Add NeMo Guardrails** at the planner boundary so off-domain prompts route to a refusal rather than calling the specialist with garbage inputs (`guardrails-on-spark`).
6. **Measure two things**: planner-only vs Eywa accuracy on a held-out domain question set (paper claims improvement on structured-data tasks), and end-to-end latency budget (planner tok/s × tokens-per-step + specialist forward-pass time).

