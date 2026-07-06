# Consistency log

Durchlaufendes Verifikationslog für Thesis–Code-Abgleich. Neue Einträge mit Datum oben anfügen.

## 2026-07-06 — Gate B Code-Readiness (abgeschlossen)

**Freeze-/Merge-Commit:** `6ca2857` (PR [#18](https://github.com/keanukf/metacog-llm-compute/pull/18) → `main`)

| Prüfpunkt | Ergebnis |
|-----------|----------|
| Volle Testsuite lokal | **OK** — `python -m pytest tests/ -v` → **250 passed**, 0 skipped, 4 warnings (`data/results/gate_b/pytest_local_merge_6ca2857.log`) |
| Volle Testsuite Pod (4090, vLLM 0.19.1) | **OK** — **250 passed**, 0 skipped, 2 warnings (`/workspace/metacog-llm-compute/data/results/gate_b/pytest_pod_merge_6ca2857.log`; Repo-Tree identisch mit `6ca2857`) |
| `steps_detail`-Persistenz (echter Smoke) | **OK** — Phase-1-Smoke `configs/dev/smoke.yaml`; Run `data/results/dev_smoke/phase1_20260706_190655/`; Fixture `tests/fixtures/episode_compact_real.json` (`ep_textworld_0_C1_0`, holdout, C1, `difficulty_tier=medium`) |
| Policy-Artefakt-Roundtrip (generiert) | **OK** — `tests/fixtures/policy_roundtrip_steps.json` (160 C1/C2-Steps, 80 holdout / 80 non-holdout); `write_threshold_artifact` → `objective_definition=step_level_proxy_v1`; `load_policy(textworld, tle_mean_entropy).stage()`: low=0.000014→C0, mid=0.000085→C0, high=0.000281→C2 |
| `nearest_position` Tie-Break | **OK** — Sekundärschlüssel `step_index` + Tests in `tests/test_thresholds_grid.py` |
| Token-Accounting (C2 / VC / Retry) | **OK** — `tests/test_token_accounting.py` |

**Smoke-Hinweise:** Policy-Fixture nur C1/C2 (kein C0 — Direct-Inference ohne Thinking-Signale für Kalibrierung). Episode-Fixture: holdout-Instanz 0, Stage C1.

## 2026-07-03 — Thesis–Code-Abgleich (rev. 2026-07-03)

| # | Prüfpunkt | Ergebnis |
|---|-----------|----------|
| 1 | `configs/experiment_core.yaml`: `vc.mode: followup`, History-Keys explizit aus, `c2.n_samples`/`sample_temperature` explizit, keine `verify_*`-Keys | **OK nach PR0** — History-Keys und VC-Defaults ergänzt; `verify_*` war bereits absent |
| 2 | `src/analysis/datasets.py`: Flattening für Step-Tabelle | **OK** — Flattening vorhanden (Zeilen 224–273); Erweiterung in PR1b |
| 3 | LM Studio `/v1/responses` Logprob-Temperatur-Skala | **Offen** — empirische Prüfung via `scripts/verify_backend_parity.py` (PR1c); Caveat in `docs/pilot.md` |
| 4 | `run_phase1.py`: `steps_detail` / `vc_detail` Persistenz | **Fixed** — minimales `steps_detail` bleibt in compact JSON erhalten |
| 5 | `hf_model_card_gate.py`: Qwen3-8B hybrid-thinking | **OK** — Gate prüft MoE/gating primär; `thinking`-Hits sind informativ, kein Auto-Reject für Qwen3-8B (Thinking erforderlich für C1) |
