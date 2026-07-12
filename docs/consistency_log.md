# Consistency log

Durchlaufendes Verifikationslog für Thesis–Code-Abgleich. Neue Einträge mit Datum oben anfügen.

## 2026-07-12 — Gate C Instrument-Validierung (Phase 0 Code bereit, Pod-Session ausstehend)

**Zweck:** Code-Voraussetzungen für Gate C (Instrument-Validierung auf 5090 + parallel vLLM) sind implementiert; Live-Läufe auf RunPod stehen noch aus.

| Bereich | Ergebnis |
|---------|----------|
| `verify_backend_parity.py` | **OK lokal (mock)** — K-Coverage + Temperatur + Batch-Invarianz (`--backend server`); `--freeze-metadata-dir` |
| `preanalysis_screen.py` | **OK** — TLE/VC-AUROC pro Domäne; CLI `python -m src.analysis.preanalysis_screen <run_dir>` |
| Dev-Configs | **OK** — `configs/dev/format_vc_probe.yaml`, `signal_smoke.yaml`, `toh_parse_probe.yaml` |
| `experiment_core.yaml` | **OK** — `max_concurrent_episodes: 1` placeholder; **Pod-Sweep vor Parity** (`measure_concurrent_throughput.py`) |
| Throughput-Sweep | **bereit** — `scripts/measure_concurrent_throughput.py`, `configs/dev/throughput_probe.yaml` |
| `signal_smoke.yaml` | **OK** — 72 Episoden, `runs_per_condition: 1`; AUROC-Probe nur Lauffähigkeit (`tle_auroc_interpretable`) |
| Phase-1-Worklist | **OK** — `compute_stages` aus `phase1`-Config (z. B. C0-only für ToH-Parse-Probe) |
| Testsuite lokal | **OK** — `python -m pytest tests/ -q` → **272 passed** |
| Pod Preflight | **bereit** — `bash scripts/instrument_validation_preflight.sh`; Ablauf in `docs/runpod.md` § Instrument validation session |

**Nächster Schritt (Pod):** Cursor SSH Remote → `/workspace/metacog-llm-compute` → Preflight → vLLM → sequenzielle Läufe unter `data/results/instrument_validation/`. Evidenz und HART-Häkchen in `blueprints/gate_p1_readiness.md` erst nach archivierten Pod-Artefakten.

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
