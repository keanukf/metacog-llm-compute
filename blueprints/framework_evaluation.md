# Lightweight agent framework evaluation (TextWorld thesis setup)

This repo intentionally uses a small, custom agent loop rather than a full agent framework.
The goal is to keep **full experimental control** over per-step prompting, memory windowing, and
signal extraction (TLE/VC) while running sequential decision tasks (e.g. TextWorld).

## What we need for the thesis

- **Per-step instrumentation**: for every environment step we record:
  - exact prompt and completion
  - action parsed from the completion
  - token-level entropy (TLE) from logprobs (action generation)
  - verbalized confidence (VC) (optional follow-up call)
  - compute cost (LM calls, tokens, wall time)
- **Memory window is an experimental variable**: if a framework auto-summarizes, truncates, or
  buffers history, it can confound results (especially in Phase 2 where allocation decisions
  depend on step-local signals).
- **Reproducibility**: given an episode trace, we must be able to re-run it deterministically
  (same prompts, same step structure, same signal extraction points).

## Why keep the custom loop (recommended)

The current design (see `src/agent/base_agent.py` and `src/agent/compute_stages.py`) is a
minimal, explicit mapping:

`observation -> step_fn(stage) -> action -> env.step(action) -> next observation`

Advantages:

- **No hidden memory behavior**: history compaction, truncation, and any pinned context are
  explicit code/config, not framework internals.
- **Exact control over LM calls**: C0/C1/C2 stages correspond to 1/2/N action calls (+ optional
  VC follow-up). This is hard to guarantee with generic agent runners.
- **Stable research artifact**: far fewer API churn risks than fast-moving agent frameworks.

## Frameworks considered (and why not migrate for this thesis)

### LangChain / LangGraph

- **Pro**: LangGraph is a clean state-machine abstraction and integrates well with observability.
- **Con**: Even if memory modules are disabled, the framework layer adds complexity, and the
  ecosystem changes quickly (version churn). For this thesis the agent is not a tool-using,
  multi-node workflow; it is a step policy over an environment.

### CrewAI

- **Mismatch**: optimized for multi-agent “crew” orchestration with roles and built-in patterns.
  Overkill for a single-policy sequential environment agent.

### Hugging Face `smolagents`

- **Mismatch**: focused on tool-calling/code-agent flows. TextWorld is parser IF with a strict
  “one command per turn” policy; the framework does not reduce risk in the critical parts
  (memory windowing + per-step signals).

### PydanticAI (and similar newer frameworks)

- **Pro**: type safety and nice ergonomics.
- **Con**: still a framework dependency with change risk; does not solve the central needs
  better than a thin local loop.

## “Best of both worlds” (optional, post-pilot)

If later we want more standardization without migrating runtime control, we can:

- keep the custom loop but represent prompts as a **message list** (`system` + turn list) rather
  than a monolithic string, while still calling the same local model wrapper.
- introduce stricter typed step-result objects (e.g. `pydantic`), reducing legacy tuple juggling.

Both options preserve experimental control while keeping the door open to future integrations.

