# Scripts catalog

All entry points live in [`scripts/`](../scripts/). Run from the **repository root** unless noted.

**Status legend**

| Status | Meaning |
|--------|---------|
| `pilot-core` | Primary pilot workflow |
| `pilot-optional` | Post-run analysis, validation, or batch helpers |
| `phase-later` | Phase 1/2 experiments (after pilot) |
| `textworld` | TextWorld dataset generation and exploration |
| `dev` | Manual play / smoke tests without full experiment |
| `cloud` | RunPod setup or result transfer |

## Python scripts

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`run_pilot.py`](../scripts/run_pilot.py) | Pilot Tests 1–6, feasibility JSON, episode outputs | Mock / HF / CUDA / LM Studio sanity | `pilot-core` |
| [`run_pilot_models.py`](../scripts/run_pilot_models.py) | Full pilot once per model id (subprocess batch) | Model shortlist on LM Studio or RunPod | `pilot-optional` |
| [`validate_pilot_outputs.py`](../scripts/validate_pilot_outputs.py) | Check pilot folder for TLE/VC/trace integrity | After a pilot run | `pilot-optional` |
| [`summarize_pilot_calibration.py`](../scripts/summarize_pilot_calibration.py) | Aggregate episode JSONs (success, TLE, VC, ECE proxy) | After pilot with episode outputs | `pilot-optional` |
| [`summarize_pilot_batch.py`](../scripts/summarize_pilot_batch.py) | Summarize `pilot_batch_*` from `run_pilot_models` | After multi-model batch | `pilot-optional` |
| [`run_c1_handoff_gate.py`](../scripts/run_c1_handoff_gate.py) | Smoke-test C1 CoT→Verify parsing on real backend | Before trusting C1 on LM Studio / CUDA | `pilot-optional` |
| [`benchmark_inference.py`](../scripts/benchmark_inference.py) | Standalone inference speed (pilot Test 1 logic) | Quick tok/s check without full pilot | `pilot-optional` |
| [`analyze_run.py`](../scripts/analyze_run.py) | Post-hoc analysis for one run folder | Inspect a completed run directory | `pilot-optional` |
| [`run_prompt_ab.py`](../scripts/run_prompt_ab.py) | A/B prompt variants on one model | RunPod prompt tuning before full batch | `pilot-optional` |
| [`hf_model_card_gate.py`](../scripts/hf_model_card_gate.py) | Read-only Hub scan for model exclusion flags | Before GPU time on RunPod | `pilot-optional` |
| [`run_phase1.py`](../scripts/run_phase1.py) | Phase 1 calibration runs with checkpointing | After pilot go/no-go | `phase-later` |
| [`run_phase2.py`](../scripts/run_phase2.py) | Phase 2 adaptive allocation runs | After Phase 1 | `phase-later` |
| [`generate_textworld_games.py`](../scripts/generate_textworld_games.py) | Generate Cooking `.z8` + metadata sidecars | Building `data/tasks/textworld/` | `textworld` |
| [`sweep_textworld_difficulty.py`](../scripts/sweep_textworld_difficulty.py) | Grid sweep + C0 evaluation per cell | Tune difficulty before final 50 games | `textworld` |
| [`build_textworld_manifest.py`](../scripts/build_textworld_manifest.py) | Build `difficulty_manifest.json` with holdout split | After final instance generation | `textworld` |
| [`play_textworld.py`](../scripts/play_textworld.py) | Interactive play for one story file | Sanity-check a generated game | `dev` |
| [`play_tower_of_hanoi.py`](../scripts/play_tower_of_hanoi.py) | Interactive ToH without a model | Verify env parsing / legality | `dev` |

## Shell scripts

| Script | Purpose | When to use | Status |
|--------|---------|-------------|--------|
| [`setup_cloud.sh`](../scripts/setup_cloud.sh) | Install pinned `requirements.txt` on pod; optional model pre-download | RunPod pod setup | `cloud` |
| [`download_runpod_results.sh`](../scripts/download_runpod_results.sh) | `scp`/rsync results from pod; auto-flattens nested `results/` | After cloud pilot | `cloud` |
| [`flatten_runpod_download.py`](../scripts/flatten_runpod_download.py) | Repair nested `results/` after manual scp | After cloud pilot download | `cloud` |
| [`probe_vllm_logprobs.py`](../scripts/probe_vllm_logprobs.py) | One-shot vLLM logprob + TLE probe | RunPod L0.1 sanity before long pilot | `cloud` |
| [`restore_cursor_plans.sh`](../scripts/restore_cursor_plans.sh) | Restore `.cursor/plans/*.plan.md` from git history (local only) | After merge removed tracked Cursor plans | `dev` |

## Related docs

- Pilot usage: [`docs/pilot.md`](pilot.md)
- RunPod workflow: [`docs/runpod.md`](runpod.md)
- TextWorld dataset: [`docs/textworld.md`](textworld.md)
- Config keys: [`configs/README.md`](../configs/README.md)
