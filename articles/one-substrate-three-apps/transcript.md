# Transcript — one-substrate-three-apps

Bridge article #8 of the shared-foundation arc. Unlike articles #1–#7 this one has no evidence/ directory — no new product is installed, no benchmark is run, no new code is written. The article exists to consolidate the foundation and declare the three-way fork.

## Source material

The content draws entirely on prior sessions:

- `~/.claude/skills/tech-writer/references/use-case-arc.md` — the source of truth for the three arcs (theses, cost profiles, article progression, state-of-the-apps closing pattern).
- `/home/nvidia/.claude/projects/-home-nvidia-nvidia-learn/memory/project_nvidia_learn_editorial.md` — the uber theme ("personal AI power user + edge AI builder") and voice.
- Handoff docs for articles #1–#7 in `handoff/`. Each one's closing section seeded a state-of-the-apps triplet that this bridge synthesizes.

## Editorial decisions

- **No new install.** The handoff for article #7 explicitly said "no new install, no new code" for the bridge. Honored.
- **Two diagrams, both earned.** The bridge is "explicitly topology-heavy" (per visualizations.md), which justifies two diagrams. The signature (`OneSubstrateThreeApps.astro`) shows the hub-and-spoke: three apps → one foundation band. The in-body fn-diagram shows cost distribution per arc (archetype #6 waterfall with phase-colored segments).
- **Cost percentages are author-held estimates, not measurements.** SB 83% query / Wiki 75% ingest / Auto 91% loop are intuitive approximations of where compute goes per arc. They communicate the *shape* of the cost space. A future polish could replace them with measured numbers from S1/W1/A1 benchmarks if the percentages drift.
- **Ran short on prose deliberately.** Target 2000–2500 words. Bridge articles that over-explain foundations readers already walked through lose energy; the discipline is to retell the thesis, not the products.
- **Anti-tradeoffs section.** Framed as "where each arc is the WEAKEST choice" rather than generic tradeoffs. Forces an honest read for the reader trying to decide; more useful than "pros and cons."
- **Invitation reads like a menu, not a quiz.** Reader picks based on cost profile — which matches the "three different answers to the same question" framing.

## What's queued next

After this article lands, the arc detector will prompt for a track choice next session. Canonical first articles per track:

- **S1** — `triton-trtllm-for-query-latency` (deployment stage, Triton + TensorRT-LLM, query-latency profile)
- **W1** — `wiki-schema-and-llm-bookkeeper` (inference stage, architecture piece; no new NVIDIA product)
- **A1** — `nemo-framework-on-spark` (training stage, NeMo Framework validation run on the Spark)

Cross-track specializations (Triton × 3, Customizer × 3, Evaluator × 3) cross-link but don't re-walk the install.
