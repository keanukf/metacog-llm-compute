# Scripts catalog

All entry points live under [`scripts/`](../scripts/), grouped into purpose-named subfolders.
Run from the **repository root** unless noted (e.g. `python scripts/experiment/run_pilot.py …`).

**Status legend**

| Status | Meaning |
|--------|---------|
| `pilot-core` | Primary pilot workflow |
| `pilot-optional` | Post-run analysis, validation, or batch helpers |
| `phase-later` | Phase 1/2 experiments (after pilot) |
| `textworld` | TextWorld dataset generation and exploration |
| `dev` | Manual play / smoke tests / diagnostics without a full experiment |
| `cloud` | RunPod setup or result transfer |
| `phase1-analysis` | Real Phase 1 confirmatory/exploratory analysis (production, not a rehearsal) |

## `scripts/experiment/` — production data-collection entry points

The front door: what actually produces the thesis data.

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`run_pilot.py`](../scripts/experiment/run_pilot.py) | Pilot Tests 1–6, feasibility JSON, episode outputs | Mock / CUDA / LM Studio sanity | `pilot-core` |
| [`run_phase1.py`](../scripts/experiment/run_phase1.py) | Phase 1 calibration runs with checkpointing | After pilot go/no-go | `phase-later` |
| [`run_phase2.py`](../scripts/experiment/run_phase2.py) | Phase 2 adaptive allocation runs | After Phase 1 | `phase-later` |

## `scripts/datasets/` — task-instance generation and interactive play

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`generate_textworld_games.py`](../scripts/datasets/generate_textworld_games.py) | Generate Cooking `.z8` + metadata sidecars | Building `data/tasks/textworld/` | `textworld` |
| [`build_textworld_manifest.py`](../scripts/datasets/build_textworld_manifest.py) | Build `difficulty_manifest.json` with holdout split | After final instance generation | `textworld` |
| [`build_toh_manifest.py`](../scripts/datasets/build_toh_manifest.py) | Build ToH `difficulty_manifest.json` (holdout split, difficulty tier by disk count) | After final ToH instance generation, mirrors `build_textworld_manifest.py` | `textworld` |
| [`play_textworld.py`](../scripts/datasets/play_textworld.py) | Interactive play for one story file | Sanity-check a generated game | `dev` |
| [`play_tower_of_hanoi.py`](../scripts/datasets/play_tower_of_hanoi.py) | Interactive ToH without a model | Verify env parsing / legality | `dev` |

## `scripts/difficulty_calibration/` — tune, sweep, verify, and freeze task difficulty

Everything for the "tune and freeze difficulty" activity (former Gate D probes + the two
difficulty sweeps).

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`sweep_textworld_difficulty.py`](../scripts/difficulty_calibration/sweep_textworld_difficulty.py) | Grid sweep + C0 evaluation per cell | Tune difficulty before final 50 games | `textworld` |
| [`sweep_toh_difficulty.py`](../scripts/difficulty_calibration/sweep_toh_difficulty.py) | ToH C0-only difficulty sweep (3 vs 4 disks) + illegal-action label distribution | Tune ToH difficulty before final instances | `dev` |
| [`difficulty_metrics.py`](../scripts/difficulty_calibration/difficulty_metrics.py) | Shared corridor/success-rate metrics library (Cap derivation, success@Cap; **no CLI/`main()`** — imported by the other calibration scripts) | Not run directly | `dev` |
| [`manifest_success_smoke.py`](../scripts/difficulty_calibration/manifest_success_smoke.py) | C0 success@Cap smoke over every instance in a frozen/candidate manifest (Hard-GO check) | Before/at manifest freeze | `dev` |
| [`capture_step_traces.py`](../scripts/difficulty_calibration/capture_step_traces.py) | Capture a handful of full per-step traces (prompt/response incl. reasoning text) via `save_step_traces` | Manual debugging of a specific stage/cell, not large sweeps | `dev` |
| [`holdout_split_descriptives.py`](../scripts/difficulty_calibration/holdout_split_descriptives.py) | Descriptive holdout-vs-non-holdout table (both domains) from frozen/candidate manifests | Sanity-check the holdout split isn't systematically different | `dev` |
| [`analyze_abort_quest_distance.py`](../scripts/difficulty_calibration/analyze_abort_quest_distance.py) | Replay a TextWorld sweep's exact grid/seeds/obs_ceiling and record quest-distance-remaining at each aborted episode's final step | Telemetry recovery when the original sweep didn't persist per-step records | `dev` |
| [`inspect_abort_last_actions.py`](../scripts/difficulty_calibration/inspect_abort_last_actions.py) | Replay the easiest TW cell (r3_i1_take-only) and print the verbatim last N actions for near-goal aborts | Manual triage of *why* near-miss episodes abort (parsing vs. looping vs. cap) | `dev` |
| [`run_textworld_feasibility_probe.py`](../scripts/difficulty_calibration/run_textworld_feasibility_probe.py) | TextWorld C0/C1/C2 feasibility diagnostic on one cell (success rate only, no signal analysis) | Is a corridor cell still C0-below-{C1,C2} after reasoning? | `dev` |
| [`run_toh_feasibility_probe.py`](../scripts/difficulty_calibration/run_toh_feasibility_probe.py) | ToH C0/C1/C2 feasibility diagnostic (companion to `run_textworld_feasibility_probe.py`; `sweep_toh_difficulty.py` only covers C0) | Resolve a ToH C0 null result via C1/C2 | `dev` |
| [`validate_textworld_candidate.py`](../scripts/difficulty_calibration/validate_textworld_candidate.py) | Re-run 2–3 corridor candidates with fresh C0 episodes at the chosen production Cap | Confirm a corridor candidate before manifest freeze | `dev` |
| [`audit_textworld_prompt_vocabulary.py`](../scripts/difficulty_calibration/audit_textworld_prompt_vocabulary.py) | CPU-only audit of command verbs/forms the TW cooking grid actually needs vs. the prompt's template action list | Diagnose action-vocabulary gaps behind a hallucination/parsing failure mode | `dev` |
| [`toh_state_diversity_probe.py`](../scripts/difficulty_calibration/toh_state_diversity_probe.py) | Probe whether C0/C1 track the actual ToH board vs. reciting a fixed move pattern (picks instances whose required first move isn't the canonical opener; concurrent traces at C0/C1) | Verify state-sensitivity, not just "same input → same output" | `dev` |
| [`toh_diversity_probe_analysis.py`](../scripts/difficulty_calibration/toh_diversity_probe_analysis.py) | Local analysis of a ToH diversity/feasibility probe run: per-stage success, TLE-AUROC vs. optimal (Hanley-McNeil CI), peg-source bias breakdown | After a `toh_state_diversity_probe.py` / feasibility run | `dev` |

## `scripts/instrument_validation/` — backend parity, throughput, logprob-invariance

Checks run before/around a real run (former Gate C tooling).

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`verify_backend_parity.py`](../scripts/instrument_validation/verify_backend_parity.py) | Backend logprob parity: K-coverage, temperature invariance, batch invariance under load (thesis §5.7); `--freeze-metadata-dir` freezes (N, eps) | Backend-parity checks before Phase 1 | `dev` |
| [`measure_concurrent_throughput.py`](../scripts/instrument_validation/measure_concurrent_throughput.py) | Compare batched-episode ep/h across candidate `max_concurrent_episodes` values | Choose production N on Pod, before `verify_backend_parity.py --backend server` | `cloud` |
| [`apply_production_n.py`](../scripts/instrument_validation/apply_production_n.py) | Write the chosen `max_concurrent_episodes` from a throughput-sweep report into a config's `execution` block | After `measure_concurrent_throughput.py` picks a value | `dev` |
| [`sweep_topk_sensitivity.py`](../scripts/instrument_validation/sweep_topk_sensitivity.py) | Recompute TLE at K ∈ {5,10,20} from stored top-K logprob sidecars | K-sensitivity check without rerunning inference | `dev` |
| [`probe_vllm_logprobs.py`](../scripts/instrument_validation/probe_vllm_logprobs.py) | One-shot vLLM logprob + TLE probe | RunPod L0.1 sanity before a long pilot | `cloud` |
| [`probe_lmstudio_thinking_toggle.py`](../scripts/instrument_validation/probe_lmstudio_thinking_toggle.py) | Compare LM Studio `/v1/responses` thinking-control payload variants (reasoning effort none/low/medium) | One-off LM Studio API probe before relying on a thinking toggle locally | `dev` |
| [`smoke_parallel.py`](../scripts/instrument_validation/smoke_parallel.py) | Parallel-execution plumbing GO/NO-GO: `EpisodeScheduler` + mock/`ServerBackend`, checks completeness/uniqueness/concurrency | Before trusting concurrent Phase 1 execution on Pod | `dev` |

## `scripts/analysis_rehearsal/` — full-pipeline dry run before real data exists

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`analysis_pipeline_rehearsal.py`](../scripts/analysis_rehearsal/analysis_pipeline_rehearsal.py) | Analysis dry run: step table → grid-search → policy artifact → `load_policy` → `cluster_bootstrap` on ΔAUROC | End-to-end analysis rehearsal on pilot/C-5 data before real Phase 1 data | `dev` |
| [`h3_power_simulation.py`](../scripts/analysis_rehearsal/h3_power_simulation.py) | Monte Carlo power simulation for the H3 signal×position_norm interaction: seeds ICC/entropy from pilot data, simulates clustered binary outcomes under the planned Phase 1 design, fits `fit_h3_model` (real GEE) per replicate, reports empirical power vs. true effect size | H3 power check; see `docs/gate_e_h3_power_simulation.md` for the report | `dev` |
| [`h2_power_simulation.py`](../scripts/analysis_rehearsal/h2_power_simulation.py) | Monte Carlo power simulation for H2 (adaptive non-inferiority + log-token superiority vs. Always-C2): seeds episode-level ICC/success-rate/token stats from real canonical Phase 1 data, token-ratio scenarios anchored to a real GPU smoke test, simulates the planned Phase 2 design (n instances × 5 runs) via `cluster_bootstrap` with `h2_paired`'s real statistic/threshold definitions | H2 power check, run same-day before the one-shot Phase 2 collection; found and triggered the `h2_paired` run-averaging fix (`docs/consistency_log.md` 2026-08-05) | `dev` |

## `scripts/phase1_analysis/` — real Phase 1 confirmatory/exploratory analysis pipeline

Each stage is independently runnable and idempotent (fixed seed, deterministic `stat_fn`s); a thin
`run_all.py` chains them for full-pipeline reproduction. See `docs/phase1_analysis_report.md` for
the archived results once the pipeline has been run against the real data.

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`stage0_build_canonical_dataset.py`](../scripts/phase1_analysis/stage0_build_canonical_dataset.py) | Selects `tower_of_hanoi` from `phase1_20260722_091125` + `textworld` from `textworld_regen_20260724` (see `src/analysis/phase1_canonical.py`), asserts frozen-design invariants, writes a content-hashed manifest every later stage reads through | First stage; run before any of the below | `phase1-analysis` |
| [`stage1_preanalysis_screen.py`](../scripts/phase1_analysis/stage1_preanalysis_screen.py) | Signal variance/VC degeneration, cluster counts, class balance by domain and position (with empty-cell flag), episode-length quartiles, real ICC (`src/analysis/icc.py`) -- diagnostic, doesn't block later stages. Also renders a full per-variable APA-7-styled descriptive codebook (`src/analysis/descriptive_stats.py`, pooled + per-compute-stage) and distribution/whisker-plot figures (`src/analysis/visualization.py::plot_signal_histograms`/`plot_signal_boxplots`/`plot_episode_length_boxplot`) | Before trusting any Stage 2+ confirmatory number | `phase1-analysis` |
| [`stage2_h1a_discrimination.py`](../scripts/phase1_analysis/stage2_h1a_discrimination.py) | H1a per domain: ΔAUROC(TLE,VC) via `cluster_bootstrap`/`delta_auroc`, Holm family A; cross-checked per domain against the independent descriptive `compare_signal_calibration` path. Also renders a bootstrap-replicate-distribution histogram per domain (`plot_bootstrap_distribution`) -- the percentile-CI assumption check | H1a confirmatory result | `phase1-analysis` |
| [`stage3_h1b_calibration.py`](../scripts/phase1_analysis/stage3_h1b_calibration.py) | H1b per domain: fits `fit_tle_calibrator` on holdout steps, evaluates ΔBrier(TLE-mapped, VC/100) on non-holdout steps via `cluster_bootstrap`, Holm family D. Also renders bootstrap-distribution histograms and TLE-mapped/VC reliability diagrams (on the same non-holdout evaluation subset) per domain | H1b confirmatory result | `phase1-analysis` |
| [`stage4_h3_temporal.py`](../scripts/phase1_analysis/stage4_h3_temporal.py) | H3 per domain per signal via `fit_h3_model`; textworld TLE+VC Holm family E (confirmatory), tower_of_hanoi exploratory only. Standard errors use statsmodels' GEE `cov_type="robust"` default | H3 confirmatory + exploratory result | `phase1-analysis` |
| [`stage5_h4_domain_modulation.py`](../scripts/phase1_analysis/stage5_h4_domain_modulation.py) | H4 single test: diff-in-diff of ΔAUROC(ToH)-ΔAUROC(TextWorld) via `cluster_bootstrap_stratified` (instances resampled within each domain, not pooled), Holm family C. Also renders a bootstrap-distribution histogram | H4 confirmatory result | `phase1-analysis` |
| [`stage6_visualizations.py`](../scripts/phase1_analysis/stage6_visualizations.py) | Renders `plot_auroc_comparison_bars`/`plot_h3_marginal_effect` (`src/analysis/visualization.py`) from the Stage 0/2/4 JSON+manifest output; the H3 marginal-effect plot now overlays real binned empirical data (via `build_h3_frame`) alongside the fitted curve as a linearity-in-logit model-fit check | Before Stage 7 | `phase1-analysis` |
| [`stage7_generate_report.py`](../scripts/phase1_analysis/stage7_generate_report.py) | Renders `docs/phase1_analysis_report.md` from every prior stage's JSON output, embeds the full Stage 1 variable codebook (all 9 tables), and copies every stage's figures (1/2/3/5/6, wherever a `figures_manifest.json` exists) into the committed `docs/figures/phase1_analysis/` | Last stage; produces the archived deliverable | `phase1-analysis` |
| [`run_all.py`](../scripts/phase1_analysis/run_all.py) | Thin sequential orchestrator chaining Stages 0-7, stops on first non-zero exit | One-shot full-pipeline reproduction | `phase1-analysis` |

## `scripts/phase2_prep/` — bridging real Phase 1 results into Phase 2 collection

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`build_threshold_artifact.py`](../scripts/phase2_prep/build_threshold_artifact.py) | Grid-searches TLE/VC allocator thresholds (theta1/theta2, `step_level_proxy_v1`) against the real Phase 1 holdout data via the Stage 0 canonical manifest, writes the frozen policy artifact `adaptive_tle`/`adaptive_vc`/`eager_style` require; warns (not fatal) on a degenerate theta1>=theta2 policy | Once, before starting real Phase 2 collection | `phase1-analysis` |

## `scripts/run_readiness/` — budget, run-hygiene, resume, and output-quality checks

Pre-real-run readiness checks (former Gate F tooling) plus the live progress watcher.

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`budget_reestimate.py`](../scripts/run_readiness/budget_reestimate.py) | Token-based Phase 1+2 GPU-hour budget re-estimate from real local data (per-domain/stage token counts × throughput), replacing the flat ep/h formula | Re-estimate budget after difficulty calibration changed episode lengths | `dev` |
| [`c1_c2_quality_probe.py`](../scripts/run_readiness/c1_c2_quality_probe.py) | C1/C2 output quality control on the real backend via the actual `run_phase1_job` + `EpisodeScheduler`: full traces, majority-vote correctness, `</think>`/parse checks | Before committing to the full Phase 1/2 run | `dev` |
| [`resume_correctness_smoke.py`](../scripts/run_readiness/resume_correctness_smoke.py) | Resume-under-concurrency hard-kill test: repeatedly SIGKILL a real `run_phase1.py` subprocess mid-run and resume, plus a write-race probe for truncated-file-treated-as-done | Confirm `--resume` correctness before long runs | `dev` |
| [`run_hygiene_preflight.py`](../scripts/run_readiness/run_hygiene_preflight.py) | Fast pod-side preflight: model on ephemeral disk, Langfuse resolvable, repo under `/workspace`, no history-truncation params active | On the pod, right before a real Phase 1/2 block | `cloud` |
| [`progress_watch.py`](../scripts/run_readiness/progress_watch.py) | Run-agnostic progress watcher: polls an output dir, counts finished episodes by (domain, stage), rough steps/min | Watch a live run locally or over SSH | `dev` |

## `scripts/pilot_analysis/` — post-run pilot analysis, validation, summaries

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`validate_pilot_outputs.py`](../scripts/pilot_analysis/validate_pilot_outputs.py) | Check pilot folder for TLE/VC/trace integrity | After a pilot run | `pilot-optional` |
| [`audit_pilot_signals.py`](../scripts/pilot_analysis/audit_pilot_signals.py) | VC/TLE by stage, C2 trace audit, feasibility summary | After RunPod download | `pilot-optional` |
| [`summarize_pilot_calibration.py`](../scripts/pilot_analysis/summarize_pilot_calibration.py) | Aggregate episode JSONs (success, TLE, VC, ECE proxy) | After pilot with episode outputs | `pilot-optional` |
| [`analyze_run.py`](../scripts/pilot_analysis/analyze_run.py) | Post-hoc analysis for one run folder | Inspect a completed run directory | `pilot-optional` |
| [`build_debug_views.py`](../scripts/pilot_analysis/build_debug_views.py) | Rebuild compact `debug_views/*.json` from existing `trace_*.jsonl` in a run directory | Regenerate human-readable views without a full rerun | `dev` |
| [`benchmark_inference.py`](../scripts/pilot_analysis/benchmark_inference.py) | Standalone inference speed (pilot Test 1 logic) | Quick tok/s check without full pilot | `pilot-optional` |
| [`run_c1_handoff_gate.py`](../scripts/pilot_analysis/run_c1_handoff_gate.py) | Quality check on single-call C1 action parsing (`parse_method` / `draft_status`, unparsed rate + Wilson CI) on a real backend | Before trusting C1 on LM Studio / CUDA | `pilot-optional` |
| [`hf_model_card_gate.py`](../scripts/pilot_analysis/hf_model_card_gate.py) | Read-only Hub scan for model exclusion flags (`--repo-id`, or `--models-file` with a user-supplied list) | Before GPU time on RunPod | `pilot-optional` |
| [`diagnose_tle_distribution.py`](../scripts/pilot_analysis/diagnose_tle_distribution.py) | TLE distribution screen (near-zero mass, spread) from a completed run's logprob sidecars | §5.4 ECDF/threshold viability check (Phase 0) | `dev` |

## `scripts/cloud/shell/` — pod setup, activation, autostop, download, preflight

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`setup_cloud.sh`](../scripts/cloud/shell/setup_cloud.sh) | Pod basics: secrets, deploy key, git pull (current branch), venv, deps, model cache | RunPod pod setup | `cloud` |
| [`pod_runtime_env.sh`](../scripts/cloud/shell/pod_runtime_env.sh) | Force HF/pip caches onto container disk (overrides RunPod `/workspace/.cache/*`) | Sourced by setup/activate | `cloud` |
| [`activate_pod_env.sh`](../scripts/cloud/shell/activate_pod_env.sh) | Source PATH/HF_HOME/RESULTS_DIR in new SSH sessions | After setup on pod | `cloud` |
| [`run_with_autostop.sh`](../scripts/cloud/shell/run_with_autostop.sh) | Run Phase 1/2 with best-effort pod stop via `runpodctl` | RunPod long runs (`STOP_POD=1`, `RUNPOD_POD_ID`) | `cloud` |
| [`download_runpod_results.sh`](../scripts/cloud/shell/download_runpod_results.sh) | `scp`/rsync results from pod; auto-flattens nested `results/` | After cloud pilot | `cloud` |
| [`instrument_validation_preflight.sh`](../scripts/cloud/shell/instrument_validation_preflight.sh) | GPU/vLLM-server sanity checks before an instrument-validation run | RunPod, right before `run_instrument_validation_after_perf.sh` | `cloud` |
| [`run_instrument_validation_after_perf.sh`](../scripts/cloud/shell/run_instrument_validation_after_perf.sh) | Runs the instrument-validation sequence against an already-running vLLM server | RunPod, after `perf/vllm-server-concurrency` work lands | `cloud` |

## `scripts/cloud/python/` — cloud helper (Python)

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`flatten_runpod_download.py`](../scripts/cloud/python/flatten_runpod_download.py) | Repair nested `results/` after manual scp | After cloud pilot download | `cloud` |

## Related docs

- Pilot usage: [`docs/pilot.md`](pilot.md)
- RunPod workflow: [`docs/runpod.md`](runpod.md)
- TextWorld dataset: [`docs/textworld.md`](textworld.md)
- Config keys: [`configs/README.md`](../configs/README.md)
