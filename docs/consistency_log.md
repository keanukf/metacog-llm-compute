# Consistency log

Durchlaufendes Verifikationslog für Thesis–Code-Abgleich. Neue Einträge mit Datum oben anfügen.

## 2026-07-03 — Thesis–Code-Abgleich (rev. 2026-07-03)

| # | Prüfpunkt | Ergebnis |
|---|-----------|----------|
| 1 | `configs/experiment_core.yaml`: `vc.mode: followup`, History-Keys explizit aus, `c2.n_samples`/`sample_temperature` explizit, keine `verify_*`-Keys | **OK nach PR0** — History-Keys und VC-Defaults ergänzt; `verify_*` war bereits absent |
| 2 | `src/analysis/datasets.py`: Flattening für Step-Tabelle | **OK** — Flattening vorhanden (Zeilen 224–273); Erweiterung in PR1b |
| 3 | LM Studio `/v1/responses` Logprob-Temperatur-Skala | **Offen** — empirische Prüfung via `scripts/verify_backend_parity.py` (PR1c); Caveat in `docs/pilot.md` |
| 4 | `run_phase1.py`: `steps_detail` / `vc_detail` Persistenz | **Fixed** — minimales `steps_detail` bleibt in compact JSON erhalten |
| 5 | `hf_model_card_gate.py`: Qwen3-8B hybrid-thinking | **OK** — Gate prüft MoE/gating primär; `thinking`-Hits sind informativ, kein Auto-Reject für Qwen3-8B (Thinking erforderlich für C1) |
