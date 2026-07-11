> **SUPERSEDED** — siehe Gate F / [`thesis_design.md`](thesis_design.md). Dieses Dokument ist als Ganzes veraltet (u. a. Qwen2.5-3B, RTX 3090, $0.22/hr, 16h-Phase-1-Schätzung, Delayed-Cue); kein line-by-line Update.

# Thesis Completion Plan v3 — March 23 to April 29, 2026

**Keanu Forthmann · M.Sc. Artificial Intelligence · IU Internationale Hochschule**
**Metacognitive Effort Allocation in Sequential Language Model Agents**

---

## Domain Change: Tower of Hanoi Replaces Delayed-Cue Recall

The second task domain is now **Tower of Hanoi** (text-based), replacing Delayed-Cue Recall. Rationale:

- **Narrative consistency:** Both domains are genuinely sequential. Delayed-Cue was a retrieval task with cosmetically sequential structure; steps were independent and decisions did not compound. Tower of Hanoi is a real sequential planning task where every move changes the state space for subsequent moves.
- **Clean H4 contrast axis:** TextWorld tests exploration under partial observability (agent doesn't know the map). Tower of Hanoi tests planning under full observability (agent sees entire state, must compute solution path). Both are sequential decision-making; the cognitive demand differs. This is a stronger contrast than memory-vs-planning, which crosses out of the thesis scope.
- **Cognitive psychology grounding:** Tower of Hanoi is one of the most studied planning tasks in cognitive psychology (executive function, problem-solving, System 2 engagement). Well-established in Kahneman's framework.
- **Implementation simplicity:** Deterministic state (three pegs, N disks), trivial legality checking, known optimal solution gives precise step-level correctness (optimal / legal-but-suboptimal / illegal). Simpler than Delayed-Cue content generators.
- **Difficulty control:** Number of disks (3 = 7 optimal moves, 4 = 15, 5 = 31) and partial starting positions allow clean targeting of 30–50% baseline success rate.

**Updated H4:** "Metacognitive signal calibration differs between exploration-based sequential tasks (TextWorld, partial observability) and planning-based sequential tasks (Tower of Hanoi, full observability)."

**Updated §5.5.2:** Tower of Hanoi, not Delayed-Cue. The domain asymmetry is now *within* sequential decision-making (exploration vs. planning), not across categories (sequential vs. retrieval).

---

## Situation Assessment

**Time budget:** 5.5 weeks × 30 hours/week = **~165 hours**
**Deliverable:** 80 pages main text (±10%, so 72–88), Arial 11pt, 1.5 spacing, Word→PDF via Turnitin
**Exposé:** Approved. Supervisor expects the design as submitted.
**Tools:** Claude Desktop/CLI/API for writing and revision, Cursor for coding, self-built literature manager

### What exists and is solid

- Literature infrastructure: 700+ screened, 200+ relevant, Zotero sync, SQLite DB
- Claim matrices: Ch2 (v3, 34 sources) and Ch3 (v1, 25 sources) structurally complete
- Outline: detailed, narratively coherent, section IDs mapped
- Experiment codebase: 21 passing tests (mocked), model wrappers (vLLM/HF/LM Studio), full agent loop with C0/C1/C2, TLE and VC signal extraction, MLflow tracking, checkpointing, Phase 1 episode loop
- Pilot script validated on M1 and LM Studio (full run_pilot.py executed)
- Design decisions locked: 2×3×2 factorial, step-level correctness, scope simplifications justified

### What is missing (ranked by criticality)

1. **Tower of Hanoi environment** — second core domain (H4), replaces the Delayed-Cue placeholder
2. **Step-level correctness labels** — required for calibration ground truth; TextWorld env needs extension, Tower of Hanoi has it by construction (optimal solution is known)
3. **Real GPU pilot** — no throughput/signal validation on target hardware (RTX 3090)
4. **Pre-experiment validation** — difficulty sweep, signal discrimination (AUROC > 0.6), action parseability check for Tower of Hanoi
5. **Phase 2 script** — allocator loop is a stub
6. **Statistical analysis code** — mixed-effects models, permutation tests, step-position interaction
7. **Chapter prose** — zero pages written
8. **Word template** — IU formatting requirements

---

## Guiding Principles

1. **Experiments on the critical path, writing is parallelizable.** GPU runs happen at night; writing happens during the day.
2. **Code with Cursor, write with Claude.** Cursor agents for implementation, Claude for prose from claim matrices and review.
3. **Pre-experiment validation is not optional.** Especially critical for Tower of Hanoi: must confirm Qwen2.5-3B produces parseable moves (>80% parseability) before committing GPU hours.
4. **Chapters 2 and 3 can be drafted now.** They depend only on claim matrices.
5. **Results chapters (6–7) and Discussion (8) need data.** Blocked until Phase 1/2 complete.

---

## Week-by-Week Plan

### WEEK 1: March 23–29 — Code Completion + GPU Pilot + Start Writing

**Goal:** Close all implementation gaps. Validate both domains on real GPU. Start Ch2 draft.

#### Monday March 23 — Tower of Hanoi Environment [Cursor]

Implement `src/environments/tower_of_hanoi.py`:

**State representation:**
- Three pegs (A, B, C), each a list of disk sizes (largest = N, smallest = 1)
- Initial state: all disks on peg A, sorted largest-to-smallest
- Goal state: all disks on peg C, sorted

**Text interface (same as TextWorldEnv):**
- `reset()` → observation describing current state and goal
- `step(action)` → next observation after applying move
- `.done` = True when goal reached or max_steps exceeded
- `.observation` = current state as text
- `.task_success` = True only if goal state reached

**Observation format example:**
```
Current state: Peg A: [3, 2] | Peg B: [1] | Peg C: []
Goal: Move all disks to Peg C.
Enter your move (e.g., 'Move disk from A to C'):
```

**Step-level correctness (three levels):**
- `"optimal"`: move matches the next step in the known optimal solution
- `"legal"`: move is legal (no larger disk on smaller) but not optimal
- `"illegal"`: move violates rules (attempted, state unchanged)

Store in `env.step_results: list[dict]` with keys: `step_index`, `action`, `correctness`, `state_before`, `state_after`.

**Optimal solution:** Pre-compute the recursive solution for the given N disks. Compare each agent move against the next expected optimal move.

**Instance generator:** `generate_instances(n, seed, num_disks_range=[3,4], allow_partial_start=True)`:
- Vary number of disks (3 or 4) across instances for within-domain difficulty variance
- Optional partial start: begin K moves into the optimal solution to reduce episode length
- Return replayable instance dicts with `id`, `num_disks`, `initial_state`, `goal_state`, `optimal_solution`, `max_steps`

**Action parsing:** Accept formats like "Move disk from A to C", "A to C", "A C", "a→c". Normalize and extract source/target peg. If unparseable, count as illegal move and return an observation asking for a valid move format.

**Unit tests:** `tests/test_07_tower_of_hanoi.py` (~12 tests):
- Instance generation (count, determinism, schema)
- Env interface (reset, step, done, observation)
- Correctness tracking (optimal, legal, illegal moves)
- Goal detection (task_success when solved)
- Difficulty scaling (3 vs. 4 disks produce different episode lengths)
- Action parsing (various formats)

#### Tuesday March 24 — Step-Level Correctness for TextWorld + Phase 2 Loop [Cursor]

**TextWorld step-level correctness:**
- Extend `TextWorldEnv` to track action validity via game engine feedback
- Real TextWorld: parse reward signal and score changes per step
- Stub mode: heuristic correctness (any non-empty action = legal)
- Store in `env.step_results` (same structure as Tower of Hanoi for consistency)

**Update `base_agent.py`:**
- After episode loop, check for `env.step_results` and attach to return dict as `step_correctness`
- This makes step-level ground truth available for calibration analysis regardless of domain

**Phase 2 episode loop:**
- Implement full loop in `run_phase2.py`: for each (domain, instance, strategy, run), create env, run adaptive episode where `allocator.allocate()` determines compute stage per step
- Wire checkpointing and MLflow tracking (same pattern as Phase 1)

**Update `_make_env` in run_phase1.py and run_phase2.py:**
- `domain == "textworld"` → `TextWorldEnv(game_file=...)`
- `domain == "tower_of_hanoi"` → `TowerOfHanoiEnv(task=instance_dict)`

#### Wednesday March 25 — Real GPU Pilot [RunPod RTX 3090]

```bash
python scripts/run_pilot.py --config configs/pilot.yaml --output-dir data/results --pilot-mode cuda
```

Standard pilot validation plus **Tower of Hanoi parseability check:**
1. tok/s ≥ 80
2. TLE discriminates easy vs. hard prompts
3. VC parseable from Qwen2.5-3B real outputs
4. TextWorld env works, agent generates valid actions
5. **Tower of Hanoi: run 20 episodes at C0 with 3-disk problems. Check: >80% of moves are parseable (even if wrong). If <80%, test with few-shot formatting examples in prompt.**
6. E2e episodes produce valid JSON with TLE/VC/step_correctness
7. Budget projections hold

**Go/No-Go decisions:**
- If Tower of Hanoi parseability < 80% even with few-shot → add action-space constraining (list valid moves in observation) and retest
- If still fails → fall back to BlocksWorld as alternative (similar structure, may be more amenable to text-based action format)
- If pilot passes ≥8/10 checks → proceed

#### Thursday March 26 — Pre-Experiment Validation [RunPod, keep pod running]

1. **TextWorld difficulty sweep:** `run_calibration.py` — target 30–50% C0 success rate
2. **Tower of Hanoi difficulty sweep:** run C0 on 3-disk and 4-disk instances, with and without partial starts. Find the configuration that yields ~30–50% task success. Record: how many moves are optimal vs. legal vs. illegal?
3. **Signal discrimination:** compute AUROC of TLE and VC as predictors of step-level correctness across both domains; threshold AUROC > 0.6
4. Generate `difficulty_manifest.json` for both domains with per-instance difficulty tiers

**If AUROC < 0.6:** adjust TLE aggregation, consider prompt tuning for VC, or reframe H1.

#### Friday March 27 — Statistical Analysis Code [Cursor]

Implement `src/analysis/comparison.py`:
- Mixed-effects model setup (`statsmodels` or `pymer4`)
- Permutation test for ECE differences (H1: TLE vs. VC)
- Step-position × signal-type interaction (H3)
- Effect size computation (Cohen's d with bootstrapped 95% CIs)
- Pairwise contrasts (Tukey-adjusted) for Phase 2 strategy comparisons

#### Saturday–Sunday March 28–29 — Writing: Ch2 §2.1–2.2 [Claude]

Draft Chapter 2, Moves 1–2 (~13 pages):

**§2.1** Language Models as Sequential Decision Systems (~5 pp):
- 2.1.1 Transformer Architecture and Token-Level Inference (claims 1–2)
- 2.1.2 The Agent Loop (claims 3–4)
- 2.1.3 The Allocation Opportunity (claims 5–8)

**§2.2** Cognitive Effort Allocation (~8 pp):
- 2.2.1 Dual-Process Theory (claims 9–10)
- 2.2.2 Expected Value of Control (claims 11–12)
- 2.2.3 Metacognitive Monitoring and Control (claims 13–16)
- 2.2.4 Temporal Degradation (claims 17–19)

Follow "name the bridge, don't cross it" in §2.2. Full bridge is §2.3 (Week 2).

**Week 1 checklist:**
- [ ] Tower of Hanoi environment implemented and tested
- [ ] Step-level correctness for TextWorld + base_agent integration
- [ ] Phase 2 episode loop runnable
- [ ] Real GPU pilot completed, Go/No-Go confirmed
- [ ] Tower of Hanoi parseability confirmed (>80%)
- [ ] Pre-experiment validation passed (both domains)
- [ ] Statistical analysis code functional
- [ ] Ch2 §2.1–2.2 draft (~13 pages)

---

### WEEK 2: March 30 – April 5 — Phase 1 Experiments + Write Ch2–5

**Goal:** Run Phase 1 calibration. Complete Ch2, Ch3, Ch4, Ch5 drafts.

#### Monday March 30

**Daytime:** Write Ch2 §2.3 The Bridge (~7 pp, claims 20–34). Map psychological constructs onto computational analogues explicitly. This is the structural contribution of Chapter 2.

**Evening ~20:00:** Start Phase 1 TextWorld overnight run (~8h):
```bash
python scripts/run_phase1.py --config configs/experiment_core.yaml --pilot-mode cuda --domain textworld --resume
```

#### Tuesday March 31

**Daytime:** Draft Ch3 §3.1–3.3 (~5 pp, claims 1–16 from Ch3 matrix). Entropy-based allocation, metacognitive probes, LLM metacognitive capabilities.

**Evening:** Start Phase 1 Tower of Hanoi overnight run (~8h).

#### Wednesday April 1

**Morning:** Download Phase 1 data. Run initial analysis: ECE, Brier, reliability diagrams, step-position calibration curves, cross-domain comparison (TextWorld vs. Tower of Hanoi).

**Afternoon:** Threshold optimization (grid search over θ₁, θ₂ on 10% validation split per domain).

**Evening:** Draft Ch3 §3.4–3.6 (~5 pp, claims 17–30). §3.6 positioning must make the gap feel inevitable.

#### Thursday April 2

Draft Ch4 (Research Questions and Hypotheses, ~4 pp): derive RQ1–RQ3 from Ch2 theory + Ch3 empirical boundaries. State H1–H4 with updated H4 framing (exploration vs. planning).

#### Friday–Sunday April 3–5

Draft Ch5 Methodology (~10 pp):
- 5.1 Research Design (2×3×2 factorial)
- 5.2 Operationalization of TLE and VC
- 5.3 Compute Stages (C0/C1/C2)
- 5.4 Adaptive Allocation Mechanism (report optimized thresholds)
- 5.5 Task Environments
  - 5.5.1 TextWorld Cooking (exploration under partial observability)
  - 5.5.2 Tower of Hanoi (planning under full observability; justify the contrast axis, discuss cognitive psychology grounding, describe step-level correctness metric with three levels)
- 5.6 Baselines (Always-C0, Always-C2, Random, EAGer-Style)
- 5.7 Model Selection (Qwen2.5-3B)
- 5.8 Statistical Analysis Plan
- 5.9 Methodological Limitations (single model, rule-based allocator, solution space compression in Tower of Hanoi as episode nears completion)

**Week 2 checklist:**
- [ ] Phase 1 complete (both domains, all conditions, data downloaded)
- [ ] Initial Phase 1 analysis and threshold optimization done
- [ ] Ch2 complete (~20 pp)
- [ ] Ch3 complete (~10 pp)
- [ ] Ch4 complete (~4 pp)
- [ ] Ch5 complete (~10 pp)

---

### WEEK 3: April 6–12 — Phase 2 Experiments + Results Chapters

**Goal:** Run Phase 2. Generate all figures. Draft Ch6 and Ch7.

#### Monday April 6

**Daytime:** Generate all Phase 1 visualizations:
- Reliability diagrams (TLE and VC, both domains)
- Step-position calibration curves (H3, primarily TextWorld, secondary Tower of Hanoi)
- TLE/VC distributions by correctness level (for Tower of Hanoi: optimal vs. legal vs. illegal)
- Cross-domain comparison plots (TextWorld vs. Tower of Hanoi calibration)

Export as high-res PNGs.

**Evening:** Start Phase 2 TextWorld overnight (~10h, 1500 episodes).

#### Tuesday April 7

**Daytime:** Draft Ch6 Results: Signal Calibration (~6 pp):
- 6.1 Overall Calibration of TLE
- 6.2 Overall Calibration of VC
- 6.3 Temporal Degradation Across Episode Steps
- 6.4 Cross-Domain Comparison (exploration vs. planning)

**Evening:** Start Phase 2 Tower of Hanoi overnight (~10h).

#### Wednesday April 8

**Morning:** Verify Phase 2 data. Rerun failed episodes if needed (buffer night run).

**Afternoon:** Full Phase 2 analysis: success rates, compute efficiency, mixed-effects model, pairwise contrasts, effect sizes, allocation pattern analysis. Compare allocation patterns across domains (does the allocator choose differently for exploration vs. planning?).

#### Thursday–Friday April 9–10

Draft Ch7 Results: Adaptive Allocation (~8 pp):
- 7.1 Performance Comparison Against Baselines
- 7.2 Compute Efficiency Analysis
- 7.3 Step-Level vs. Prompt-Level Allocation (EAGer-Style Comparison)
- 7.4 Allocation Patterns: When Does the Agent Choose to Deliberate?

#### Saturday–Sunday April 11–12

Finalize all figures and tables. §3.6 positioning table. Colorblind-friendly, properly captioned.

**Week 3 checklist:**
- [ ] Phase 2 complete (both domains, all strategies)
- [ ] Full statistical analysis done
- [ ] All figures and tables generated
- [ ] Ch6 complete (~6 pp)
- [ ] Ch7 complete (~8 pp)

---

### WEEK 4: April 13–19 — Discussion + Introduction + First Revision

**Goal:** Complete all chapter drafts. First full revision.

#### Monday–Tuesday April 13–14 — Ch8 Discussion (~10 pp)

- 8.1 Interpretation Through Dual-Process Theory — does the allocator function as an EVC-informed switch?
- 8.2 Signal Quality: Sequential vs. Single-Turn — how do results compare to EAGer/MeCo single-turn findings?
- 8.3 Temporal Degradation — distinguish signal-level noise from calibration-level noise; connect to Efklides (2006), Liu et al. (2023)
- 8.4 Exploration vs. Planning — do metacognitive signals calibrate differently across the two sequential domains? What does this tell us about domain-specificity of entropy-based self-monitoring?
- 8.5 Limitations (proactive: single model, rule-based allocator, Tower of Hanoi solution space compression, limited runs)
- 8.6 Ethical Considerations

#### Wednesday April 15 — Ch1 Introduction (~4 pp)

- 1.1 The Compute Allocation Problem
- 1.2 Research Gap: Single-Turn to Sequential
- 1.3 Structure of the Thesis

#### Thursday April 16 — Ch9 Conclusion + Future Work (~3 pp)

- 9.1 Summary of Contributions
- 9.2 Future Research (learned allocator, more models, longer episodes, additional sequential domains)
- 9.3 Toward Metacognition-Aware AI Agents

#### Friday–Sunday April 17–19 — First Full Revision Pass

Read entire thesis end-to-end. Check: argument coherence, precision, forbidden constructions, interdisciplinary coherence, cross-references, citation completeness. Verify that "Delayed-Cue" appears nowhere in the final text.

**Week 4 checklist:**
- [ ] Ch8 complete (~10 pp)
- [ ] Ch1 complete (~4 pp)
- [ ] Ch9 complete (~3 pp)
- [ ] First revision pass done
- [ ] Full manuscript assembled (~78 pp)

---

### WEEK 5: April 20–29 — Polish, Format, Submit

#### Monday–Tuesday April 20–21 — Second Revision + APA Check

Style revision, APA 7 citation check (every in-text ↔ bibliography), abstract (~300 words).

#### Wednesday April 22 — Word Formatting

IU-compliant Word document: title page, Roman/Arabic page numbers, Arial 11pt/1.5/block text/2cm margins, max 3 heading levels, figure/table lists, abbreviation list, Eidesstattliche Erklärung.

#### Thursday April 23 — Final Proofread

Final read. Check figures at print resolution. Verify Turnitin access.

#### Friday–Sunday April 24–26 — Buffer

If on track: Kolloquium preparation. If behind: catch-up.

#### By April 29, 23:59 — Submit

PDF via Turnitin. Notify supervisor.

**Week 5 checklist:**
- [ ] APA 7 check complete
- [ ] Abstract written
- [ ] Word document formatted per IU
- [ ] Eidesstattliche Erklärung signed
- [ ] PDF ≤ 100MB, ≤ 30% images
- [ ] Submitted via Turnitin

---

## Hour Budget

| Activity | Hours |
|----------|-------|
| Code: Tower of Hanoi env, step correctness, Phase 2 loop | 10 |
| Code: Statistical analysis + visualization | 8 |
| GPU: Pilot + validation + Phase 1/2 (active time) | 12 |
| Analysis: data processing, plotting | 10 |
| Writing: Ch2 (~20 pp) | 16 |
| Writing: Ch3 (~10 pp) | 10 |
| Writing: Ch4 (~4 pp) | 4 |
| Writing: Ch5 (~10 pp) | 10 |
| Writing: Ch6 (~6 pp) | 6 |
| Writing: Ch7 (~8 pp) | 8 |
| Writing: Ch8 (~10 pp) | 12 |
| Writing: Ch1 + Ch9 (~7 pp) | 7 |
| Revision: two full passes | 14 |
| Formatting + front/back matter | 14 |
| Buffer | 24 |
| **Total** | **~165** |

---

## Decision Points

| When | Condition | Action |
|------|-----------|--------|
| March 25 (GPU pilot) | ≥8/10 checks pass | Proceed |
| March 25 | tok/s < 80 | Scale GPU budget +$5–10 |
| March 25 | Tower of Hanoi parseability < 80% | Add few-shot examples; if still fails, add valid-move list to observations; if still fails, fall back to BlocksWorld |
| March 26 | AUROC < 0.6 for both signals | Emergency: debug signals before Phase 1 |
| April 1 (Phase 1 data) | ECE > 0.4 for both signals | H1 null result; reframe Discussion |
| April 1 | No temporal degradation | H3 null result; valid if reported fully |
| April 8 (Phase 2 data) | Adaptive shows no advantage | H2 null; "SLMs lack sufficient metacognitive signal quality" |
| April 19 (final week) | Any chapter undrafted | Prioritize missing chapter over revision |

---

## Page Budget

| Chapter | Target Pages | Depends On |
|---------|-------------|------------|
| 1. Introduction | 4 | Written late (Week 4) |
| 2. Theoretical Background | 20 | Claim matrix (ready) |
| 3. Related Work | 10 | Claim matrix (ready) |
| 4. RQ and Hypotheses | 4 | Ch2 + Ch3 |
| 5. Methodology | 10 | Design (ready) |
| 6. Results: Signal Calibration | 6 | Phase 1 data (Week 2) |
| 7. Results: Adaptive Allocation | 8 | Phase 2 data (Week 3) |
| 8. Discussion | 10 | Results (Week 3) |
| 9. Conclusion | 3 | Last |
| **Total** | **~75–78** | **Target: 72–88** |

---

## Tower of Hanoi: Implementation Spec Summary

**File:** `src/environments/tower_of_hanoi.py`

**State:** Three pegs (lists of ints). Disks numbered 1 (smallest) to N (largest).

**Observation format:**
```
Current state: Peg A: [3, 2] | Peg B: [1] | Peg C: []
Goal: Move all disks to Peg C.
Valid moves: A→B, A→C, B→A, B→C
Enter your move (e.g., 'Move disk from A to C'):
```

**Step-level correctness:**
- `"optimal"` — matches next move in pre-computed optimal solution
- `"legal"` — legal but not optimal (state still changes)
- `"illegal"` — violates rules (state unchanged, observation repeats with error message)

**Difficulty knobs:**
- `num_disks`: 3 (7 optimal moves) or 4 (15 optimal moves)
- `partial_start`: begin K moves into optimal solution (reduces required moves)
- `max_steps`: cap to prevent infinite loops

**Instance generator:** `generate_instances(n, seed, num_disks_range=[3,4])` returns list of dicts with `id`, `num_disks`, `initial_state`, `goal_state`, `optimal_solution`, `max_steps`.

**Task success:** `True` only if goal state reached within max_steps.

**Parseability:** Accept "Move disk from A to C", "A to C", "A C", "a→c", "A->C". Normalize to (source_peg, target_peg). Unparseable input = illegal move.

**Config addition to `experiment_core.yaml`:**
```yaml
tower_of_hanoi:
  num_disks_range: [3, 4]
  allow_partial_start: true
  partial_start_moves: [0, 3]  # start 0–3 moves into optimal solution
```

**Cascading changes:**
- `configs/experiment_core.yaml`: replace `delayed_cue` with `tower_of_hanoi` in phase1/phase2 domains
- `scripts/run_phase1.py` and `run_phase2.py`: update `_make_env` for `tower_of_hanoi` domain
- `chapters/outline.md`: update §5.5.2 title and focus
- Claim matrices: no changes needed (Ch2/Ch3 don't reference Delayed-Cue directly)
