# Consistency log

Durchlaufendes Verifikationslog für Thesis–Code-Abgleich. Neue Einträge mit Datum oben anfügen.

## 2026-07-16 — Post-Fix-Sweeps (Pod, 4ac4431): Korridor-Kandidaten, ToH-Nullbefund, C1>C2

**Kontext:** Erste echte Sweeps auf dem korrekt verdrahteten Code (siehe Eintrag unten), 5090-Pod,
`--real`, alle mit `--config configs/dev/gate_d_calibration.yaml` bzw. `gate_d_diagnostic.yaml`.

### TextWorld C0 Corridor-Sweep (27 Zellen × 4 Instanzen, `--runtime-max-steps 45`)

`data/results/gate_d_calibration/textworld_sweep/sweep_results.json`. Production-Cap-Ableitung: 45
(≈ obs_ceiling). **WARNUNG aus dem Script selbst:** Trunkierungsrate 68,5 % (Schwelle 5 %) → p90
potenziell downward-biased. Top-Kandidaten (success@Cap=0.500): `r5_i1_take+cook`,
`r3_i2_take-only`, `r5_i3_take-only`.

**Kontrollmessung mit Cap 70** (`data/results/gate_d_calibration/textworld_sweep_cap70/`,
gleiche Instanzen/Seed): Trunkierungsrate **steigt** leicht auf 70,4 % statt zu sinken.
**Schlussfolgerung: kein Floor-Effekt** — gescheiterte Episoden hängen strukturell fest
(Loops/Halluzinationen), nicht Step-Budget-limitiert. Höherer Cap ändert am Bild nichts;
45 reicht für die Korridor-Entscheidung, muss nicht weiter erhöht werden. Neue Top-Kandidaten bei
Cap 70 (success@Cap=0.500): `r5_i3_take-only`, `r3_i1_take+cut+cook`, `r7_i2_take-only` — teils
andere Zellen als bei Cap 45, Kandidatenlage also nicht über beide Läufe stabil (n=4/Zelle ist klein
und verrauscht; vor Freeze ggf. n erhöhen für die engere Auswahl).

### Tower of Hanoi C0-Sweep (`--instances-per-combo 10`)

`data/results/gate_d_calibration/toh_sweep/sweep_results.json`. **0,0 % Erfolg bei 3 UND 4 Disks**,
Illegal-Rate 75,9 % (3 Disks) bzw. 89,3 % (4 Disks). Deutlich schlechter als TextWorld C0 und noch
nicht eingeordnet, ob das echte Schwierigkeit oder eine ToH-spezifische Prompt-/Parsing-Lücke ist
(anders als TextWorld hat ToH kein Reasoning-Vokabular-Problem, aber ggf. andere Parsing-Eigenheiten
der `A->C`-Ausgabe). **Offen — braucht C1/C2-Vergleich, den es für ToH noch nicht gibt** (nur
`sweep_toh_difficulty.py`, C0-only by design).

### TW C0/C1/C2-Feasibility (r3_i1_take-only, n=8, Cap 45)

`data/results/gate_d_diagnostic/feasibility/feasibility_report.json`. **C0: 37,5 % (3/8)** — liegt
im 30–50-%-Zielkorridor. **C1: 100 % (8/8)** — deutlich über Korridor, Reasoning löst diese Zelle
fast immer. **C2: 50 % (4/8)** — auffällig **niedriger als C1** trotz mehr Compute
(Self-Consistency, N=3). Erste direkte Post-Fix-Evidenz für einen genuinen C1>C2-Befund, nicht durch
den Wiring-Bug erklärbar (der ist ja gefixt).

### Trace-Probe (6 Episoden, volle Token-Kette inkl. Reasoning-Text)

`data/results/gate_d_calibration/trace_probe/` (`trace_*.json` + `traces/trace_*.jsonl` mit
komplettem Prompt/Response pro Schritt, inkl. `<think>`-Blöcken). Ergebnis: TW-C1 1/2 (Sieg bei 39
von 45 Schritten — knapp), TW-C2 0/2, ToH-C0 0/2. Bestätigt den C1>C2-Befund nochmal auf kleiner
Stichprobe. **Diese Traces sind für eine manuelle Reasoning-Text-Analyse (warum verliert C2 gegen
C1? was passiert bei den ToH-Fehlschlägen konkret?) noch nicht ausgewertet — nur die
Erfolg/Miss-Zahlen liegen vor.**

### Offene Fragen für nach der Session (keine autonome Entscheidung getroffen)

- ToH C0-Nullbefund: braucht C1/C2-Vergleich, ggf. leichtere Konfigurationen (partial_start) vor
  einer Domain-Viability-Entscheidung.
- TW-Korridor-Kandidaten sind zwischen Cap-45- und Cap-70-Lauf nicht stabil — vor Manifest-Freeze
  mit größerem n (statt 4) bestätigen.
- Der C1>C2-Befund (auch strukturell durch die Traces bestätigt) ist eine echte, interessante
  Eigenschaft dieser Aufgabe/dieses Modells — noch nicht durch Reasoning-Text-Lektüre erklärt.
- Inventar-Beobachtbarkeits-Frage (siehe oben, 2026-07-16 Parser-Audit-Kontext) weiterhin
  unangetastet, wartet auf explizites Go.

## 2026-07-16 — Gate D diagnostic scripts: missing config wiring silently capped C1/C2 at ~128 tokens

**Problem:** Sechs Gate-D-/Sweep-Dev-Scripts riefen `get_step_fn(stage)` ohne
`resolve_step_fn_kwargs(config, domain)` auf. Ohne die aufgelösten Kwargs fiel
`domain_prompts.<domain>.prefix` auf einen leeren Prompt zurück, und `cot_max_tokens`
fiel von den konfigurierten 8192 auf den internen Nofallback `max(128, action_max_tokens*2)`
≈ 128 Token — für C1/C2 (natives Thinking) faktisch unbrauchbar.

**Betroffen:** `run_gate_d_feasibility.py`, `inspect_gate_d_abort_actions.py`,
`analyze_gate_d_abort_distance.py`, `gate_d_manifest_smoke.py`,
`sweep_textworld_difficulty.py` (der ursprüngliche 216-Episoden-Sweep selbst),
`sweep_toh_difficulty.py`. **Nicht betroffen:** `src/execution/episode_runner.py`
(Phase-1/2-Produktionspfad) und `run_c1_handoff_gate.py` — beide bereits korrekt verdrahtet.

**Live-Beleg (Pod, `r3_i1_take-only`, 2 Instanzen):** Vor dem Fix brach C1 bei jedem
Step auf die wörtliche Aktion `"<think>"` ab, C2 auf eine leere Aktion — 0/6 Erfolge über
C0/C1/C2. Nach dem Fix (`prompt_prefix` nicht-leer, `c1_cot_max_tokens=c2_cot_max_tokens=8192`):
2/6 echte Siege (C1 in 16 Schritten, C2 in 23), beide über `prepare meal → eat meal` mit
korrekter `won`/`done`-Terminierung. Kein Terminierungs-Bug — der frühere Verdacht dazu ist
damit ebenfalls entkräftet.

**Fix:** Alle sechs Scripts nutzen jetzt `resolve_step_fn_kwargs(config, domain)` +
neue Konstante `HISTORY_CFG_KEYS` (`src/utils/step_config.py`) statt eines pro Script
duplizierten Key-Sets. Scripts sind gefixt, aber **noch nicht neu laufen gelassen**
(zeitlicher Aufwand, Entscheidung steht aus).

**Konsequenz — Baseline-Hygiene:** Alle vor diesem Fix erzeugten Gate-D-Ergebnisse
(`feasibility_report.json`, `textworld_sweep/sweep_results.json`,
`textworld_abort_distance/`, `textworld_abort_action_inspection*/`) spiegeln den
~128-Token-Deckel bei C1/C2 wider und dürfen **nicht** zur Beurteilung von C1/C2-Schwierigkeit
oder -Machbarkeit herangezogen werden. Das gilt zusätzlich zur bereits bekannten
Vokabular-Baseline-Einschränkung vom 2026-07-15.

**Nächster Schritt:** Sweeps mit den gefixten Scripts neu laufen lassen, sobald Zeit/Scope
entschieden ist.

## 2026-07-15 — FREEZE-REVIEW: TextWorld static prompt vocabulary completeness

**Problem:** Die statische TextWorld-Template-Liste in `domain_prompts.textworld.prefix` deklarierte eine geschlossene erlaubte Kommandomenge (`Use only parser commands from the templates below` / `valid forms`), schloss aber lösungsnotwendige Aktionsklassen aus: Finish-Sequenz (`prepare meal`, `eat meal`) und cut-Varianten (`chop`/`slice`/`dice [object] with [tool]`). Das konfundiert Instruktionsbefolgung mit Aufgaben-Misserfolg.

**Beleg:** Generator-Walkthrough-Audit über 27 Rasterzellen (`scripts/audit_textworld_prompt_vocabulary.py`, `data/results/gate_d_calibration/textworld_vocab_audit.json`): `prepare meal` und `eat meal` in 27/27; cut-Varianten in cut-Zellen.

**Fix:** Template-Liste um genau diese fünf Formen ergänzt (`configs/experiment_core.yaml`), identisch über alle Zellen. `close [container]`, `look`, `put [object] in [container]` bleiben (legal, ggf. auf anderen Ziehungen lösungsnotwendig).

**Abgrenzung (DV-Schutz):**
- Kein admissible-commands-Einschluss; `include_admissible_commands: false` unverändert.
- Generative Aktionswahl unverändert — statische Verb-Grammatik, keine zustandsabhängige Liste.
- `fry`/`roast`/`grill` nicht im Walkthrough (nur `cook`); halluziniertes `fry` bleibt Planungsfehler in der DV.
- Synonym-Fix (`look inventory` → `inventory` wenn admissible) separat dokumentiert (Eintrag Parser-Synonym).

**Bewusst nicht im statischen Vokabular:** `put [object] on [surface]`, `insert [object] into [container]` — admissible-only, nicht lösungsnotwendig; werden als **legal** gelabelt, falls das Modell sie trifft.

**Baseline-Hygiene:** Erster TW-Sweep (216 Ep.) und Kontrollmessungen auf unvollständigem Vokabular — **nicht** als Schwierigkeits-Baseline verwenden. Belastbare Baseline erst nach Vokabular-Fix + Diagnostik-Lauf (`data/results/gate_d_diagnostic/feasibility/`).

## 2026-07-15 — Gate D TextWorld parser synonym fix (pre–sweep v2)

**Entscheidung:** Korrektur des Aktions-Parsings für **Synonyme legaler Kommandos** in TextWorld (`textworld_env.py`). Das Modell-Output (`action_raw`) bleibt unverändert; ausgeführt wird die kanonische Form nur wenn sie pre-step admissible ist.

**Erlaubt (Kategorie A, Audit `docs/gate_d_parser_audit.md`):**
- `look inventory` → `inventory`
- `check inventory` → `inventory` (gleiche Klasse, präventiv)

**Explizit nicht geändert (DV-Schutz):**
- Keine admissible commands im Prompt/Observation (`include_admissible_commands: false` unverändert)
- Kein Umlenken von Planungsfehlern (`fry …` in take-only, falsche `go *`-Richtungen)
- Keine zell- oder domain-spezifischen Prompt-Unterschiede; ToH unberührt

**Operationalisierung:** Änderung besteht den Test „Synonym legaler Absicht vs. Hilfe bei Aktionswahl". Kontrollmessung: 8× `r3_i1_take-only` nach Fix (Gate D branch).

**Kontrollmessung (2026-07-15):** Instanz 7 — `look inventory` jetzt `legal` (Synonym greift); Gesamt **0/8** Erfolg (Varianz vs. Pre-Replay 1/8). Label-Basis sauberer; Korridor weiter offen → Planungsbefund B/C.

**Nächster Schritt:** Schwierigkeitskalibrierung auf sauberer Label-Basis (moderate Decke 30–35, nicht 50); ggf. cut/cook-Vereinfachung erst nach erneuter Baseline.

## 2026-07-14 — Logprob sidecar policy (pre–Gate D)

**Entscheidung (Gate F Run-Hygiene, revidiert):** Kein binäres Sidecar an/aus. Drei Modi via `logging.logprob_sidecar_mode`:

| Modus | Inhalt | Phase 1/2 |
|-------|--------|-------------|
| `off` | Keine Sidecars | — |
| `action_window` | Top-K nur committed-action-Tokens | **Produktion (Default)** |
| `full` | Volle Completion inkl. Reasoning | Explorative Teilstichprobe |

**Produktion (`experiment_core.yaml`):** `logprob_sidecar_mode: action_window`. **Reasoning-Ausnahme:** `logprob_sidecar_full_instances: {textworld: [1,2,3], tower_of_hanoi: [1,2,3]}` — **Nicht-Holdout** (mod-10 Holdout = 0,10,20,30,40), Kap.-9-Ausblick auf konfirmatorischer Datenbasis.

**Legacy:** `logging.save_logprob_distributions` → **ValueError** (kein stiller Fallback auf `full`).

**Speicher-Hochrechnung (revidiert, aus 105004 ~116 MB/ep Vollsidecar):**

| Modus | MB/ep (Ø) | P1+P2 (4.5k ep) | gzip (×5–10) |
|-------|----------:|----------------:|-------------:|
| `off` | ~0.02 (nur Episode-JSON) | ~0.1 GB | — |
| `action_window` | ~2–3 (≈2–3 % von Voll) | **~10–15 GB** | **~1–3 GB** |
| `full` (alle Ep.) | ~116 | ~522 GB | — |
| `full` (Inst. 0, beide Dom., P1+P2 Schätzung) | — | **~few GB** | — |

**Code:** `src/utils/logprob_sidecar.py`; Sidecar-JSON-Feld `sidecar_scope`; Filter via `slice_action_logprob_tokens`.

**Nächster Schritt:** Gate D (nach Commit dieser Policy).

## 2026-07-14 — Gate C merge + Pod main (`9f8dafd`)

**Merge:** PR #21 → `main` @ `9f8dafd` (22 Commits: Gate C instrument validation, stage-wise ECDF, legacy ECDF hardening, vLLM concurrency).

**Pod:** `213.173.103.203:46239` — `git checkout main && pull` → `9f8dafd`; **`317 passed`** pytest (14.4s, venv `/root/venv-metacog`).

**Gate C:** **Done** (final, auf `main` verankert). Evidenz: `docs/instrument_validation_session.md`, `docs/freeze_review_5_4_stage_wise_ecdf.md`, C-5 `phase1_20260714_105004`.

**Gate F — Budget (C-5 ep/h, `@20` steps, N=32):**

| Item | Episodes | ep/h | Wall (h) |
|------|---------:|-----:|---------:|
| Phase 1 | 1,500 | 93 | ~16.1 |
| Phase 2 | 3,000 | 93 | ~32.3 |
| **P1+P2** | **4,500** | **93** | **~48.4** |

Quelle: `105004` — 72 ep, 47m 42s wall. Kurz-Sweep `@8` steps: ~192 ep/h (`throughput_sweep_post_perf.json`) — nur Planungs-Obergrenze, nicht Phase-1/2-Extrapolation.

**Gate F — Speicher (Sidecars, `105004`):**

| Modus | MB/ep (Ø) | P1 (1.5k) | P2 (3k) | P1+P2 |
|-------|----------:|----------:|--------:|------:|
| Episode-JSON only (`save_logprob_distributions: false`, Produktion) | ~0.02 | ~0.03 GB | ~0.06 GB | **~0.1 GB** |
| Sidecars ON (Smoke/C-6, volle Reasoning-Logprobs) | ~116 | ~174 GB | ~348 GB | **~522 GB** |

C1/C2 dominieren (TW C2 ~154 MB/ep, ToH C2 ~385 MB/ep); C0 ~0.1–0.2 MB/ep. **Entscheidung (superseded):** siehe Eintrag oben — `action_window` + Inst.-0-`full`-Subset.

**Nächster Schritt:** Gate **D** (TW/ToH Difficulty-Sweep + Manifeste), **E** (Analyse-Rehearsal E2E), **F** (Resume-Test, Run-Hygiene, Top-up vs ~48 h GPU).

## 2026-07-14 — Gate C close: legacy ECDF hardening, sidecar verification, ECDF occupancy

**P1 — Legacy `ecdf_ref` Ladepfad:** Stilles Replizieren der gepoolten ECDF auf alle Stufen **entfernt**. Artefakte ohne `ecdf_by_stage` → `ValueError` (Default `allow_legacy_pooled_ecdf=False` in `load_policy`). Pilot-only: `load_policy_pilot()` setzt explizit `True` + `UserWarning`.

**P2 — Reasoning-Logprobs in Sidecars (105004, pod-verifiziert):** **Ja, bereits enthalten** — nicht nur Action-Tokens.
- C1: `steps[].logprob_tokens` = volle Completion (Schema v1). Beispiel `ep_textworld_0_C1_0`: 177 Tokens, ~173 im Thinking-Block, ~4 Action; first token ``.
- C2: `steps[].samples[].logprob_tokens` = volle Completion pro Sample (Schema v2); gleiches Muster (~134–172 think / ~4 action).
- **Kein Sidecar-Schema-Change** (Daten bereits da; TLE-Produktionspfad unverändert).
- Größenordnung: C1-Sidecar ~55 MB vs C0 ~103 KB/Episode (Reasoning dominiert Speicher — bereits so persistiert).

**P3 — ECDF-Besetzung Holdout (105004, Instanzen 0–4, first-n):** TLE `mean_entropy` pro Domäne/Stufe:

| Domain | Stage | n | distinct | min Δpercentile | median Δpercentile |
|--------|-------|--:|---------:|----------------:|-------------------:|
| TextWorld | C0 | 84 | 84 | 0.0119 | 0.0119 |
| TextWorld | C1 | 100 | 100 | 0.0100 | 0.0100 |
| TextWorld | C2 | 100 | 100 | 0.0100 | 0.0100 |
| ToH | C0 | 100 | 100 | 0.0100 | 0.0100 |
| ToH | C1 | 98 | 98 | 0.0102 | 0.0102 |
| ToH | C2 | 90 | 90 | 0.0111 | 0.0111 |

Pilot-Smoke: ~84–100 Holdout-Steps/Stufe/Domäne (5 Inst × 3 Stages × ~6–7 Steps); Perzentil-Raster ~0.01 (≈100 Stützstellen). Kein Redesign — §5.9 Holdout-Limitationspunkt präzisiert.

**Gate C:** **Done** (final).

## 2026-07-14 — Stage-wise ECDF allocator (§5.4 design fix)

**Entscheidung:** Stufenweise ECDF implementiert (`ecdf_by_stage`: C0/C1/C2 je Domain/Signal auf Phase-1-Holdout). θ₁/θ₂ unverändert auf Perzentilskala; Grid-Search unverändert bis auf stufenpassende ECDF pro Holdout-Row/`compute_stage`. Runtime: Perzentil des Signals aus Schritt *t* gegen ECDF der Stufe, in der Schritt *t* lief (`signal_source_stage`).

**Begründung (mechanistisch, präregistrierungssauber):** Nach Reasoning-Trace kollabiert die Verteilung über die committed action — TLE misst in C1/C2 Verbalisierungssicherheit, in C0 Entscheidungssicherheit. Gepoolte ECDF verletzt stufenübergreifende Vergleichbarkeit (§5.4). Die Entscheidung stützt sich auf die **beobachtete Entropieverteilung** (Deskriptivstatistik) und Messtheorie, **nicht** auf Signal-Korrektheits-Zusammenhänge oder AUROC-Werte. `raw_logprobs`-Temperatur-Invarianz deckt die Verschiebung nicht ab (Entstehung im Reasoning, nicht Decoding).

**Kein „A oder B":** Stufenweise ECDF ist Allocator-Konstruktion, damit ein negatives H2-Ergebnis ein Signalbefund bleibt und kein Artefakt kaputter Schwellenkalibrierung.

**Deskriptive Pilot-Zahlen (105004, nicht zur Designbegründung):**
- Within-stage TLE-AUROC: Signal/Ordnung in jeder Stufe erhalten; `n_positive` 19–28 → CI ~±0.10; keine feinaufgelösten Stufeneffekte.
- TW C0 TLE 0.534 **nicht** als „near chance" gesichert.
- **H1a-Spannung (Ergebnis, keine Designänderung):** VC > TLE bei C0 (TW 0.607 vs 0.534; ToH 0.790 vs 0.678).

**Offen (§5.9):** ToH-C2 unter `legal_or_optimal`: `n_neg=1` auf 72 Pilot-Episoden — Sensitivitätsarm ggf. degeneriert; mit voller Phase-1-Instanzzahl prüfen.

**Artefakt:** `docs/freeze_review_5_4_stage_wise_ecdf.md`

**Gate C:** Allocator-Blocker **behoben** (Code). **K=20 re-eingefroren** (C-6 reconciled, unabhängig von ECDF). Gate C **Done** (313 pytest grün).

## 2026-07-14 — Within-stage AUROC + Allocator-Blocker (`105004`)

**Zweck:** Allocator-Konstruierbarkeit — gepoolte ECDF vs stufenweise ECDF (Designfix, kein empirisches „Wählen").

**Korrektheitsrate (optimal):** TW ~11% alle Stufen; ToH C0 7.9% → C1 **43.4%** → C2 32.7%.

**Within-stage AUROC (`optimal_only`, TLE / VC):**

| Domain | Stage | n | n+ | TLE | VC |
|--------|-------|--:|---:|----:|---:|
| TW | C0 | 223 | 25 | 0.534 | 0.607 |
| TW | C1 | 240 | 28 | 0.715 | 0.620 |
| TW | C2 | 240 | 28 | 0.674 | 0.599 |
| TW pooled | | 703 | 81 | 0.619 | 0.611 |
| ToH | C0 | 240 | 19 | 0.678 | 0.790 |
| ToH | C1 | 205 | 89 | 0.634 | 0.572 |
| ToH | C2 | 214 | 70 | 0.594 | 0.691 |
| ToH pooled | | 659 | 178 | **0.744** | 0.701 |

**Befund:** ToH pooled 0.744 vs C2 within 0.594 — Pooling-Artefakt plausibel. Magnitude kollabiert in C1/C2 (Median ~1e-6), Ordnung erhalten → stufenweise ECDF als minimale Korrektur.

**Gate C:** War offen (K-Freeze suspendiert). → Siehe Eintrag oben (stage-wise ECDF).

## 2026-07-14 — Gate C close: C-6 reconciled, K=20 frozen, §5.4 TLE screen

**Commit:** `bc0ef84` @ Pod — AUROC-Alignment + Regressionstest + Diagnose.

**Blockierend 1 (Produktionspfad):** Regressionstest `test_c1_production_tle_excludes_thinking_first_token_regression` — beweist, dass `extract_action_tle_from_response` nur Action-Tokens nach `` mittelt; fehlschlägt bei First-Token-/Full-Sequence-Bug. Pod: grün.

**C-6 (gefixter Sweep, Pod, volle 72 Sidecars, `105004`):**

| Domain | K=5 | K=10 | K=20 | n | Δ vs H1a |
|--------|----:|-----:|-----:|--:|---------:|
| TextWorld | 0.618 | 0.618 | **0.619** | 703 | 0 |
| ToH | 0.744 | 0.744 | **0.744** | 659 | 0 |

K stabil (Δ≤0.001). **K=20 eingefroren** (re-confirmed after stage-wise ECDF allocator fix).

**§5.4 TLE-Verteilung (`diagnose_tle_distribution.py`, `105004/tle_distribution.json`):**

| Gruppe | n | median | pct < 1e-3 | distinct |
|--------|--:|-------:|-----------:|---------:|
| TW C0 | 224 | 0.021 | 17% | 182 |
| TW C1 | 240 | 1.5e-6 | **97%** | 240 |
| TW C2 | 240 | 5.1e-7 | **95%** | 240 |
| ToH C0 | 240 | 0.066 | 19% | 227 |
| ToH C1 | 205 | 6.7e-7 | **99.5%** | 205 |
| ToH C2 | 214 | 1.7e-6 | **99%** | 214 |
| pooled | 1363 | 3.9e-6 | 70% | 1308 |

**Diagnose (kein Redesign):** C1/C2-TLE-Werte liegen fast ausschließlich unter 1e-3 bit (Median ~1e-6), aber sind **float-distinkt**. AUROC funktioniert rangbasiert. **§5.4-Fix:** stufenweise ECDF — siehe `docs/freeze_review_5_4_stage_wise_ecdf.md`.

**N-Konsistenz:** N=32, eps=0.05 — unverändert bestätigt.

**Gate C:** **Done** (C-6 + stage-wise ECDF allocator).

## 2026-07-14 — AUROC reconciliation (C-6 technical fix)

| Pfad | TW `105004` | ToH `105004` | Ursache |
|------|------------:|-------------:|---------|
| `preanalysis_screen` / `signal_discrimination_report` (alt) | 0.381 raw | 0.256 raw | Rohentropie ohne Vorzeichenflip (−entropy = H1a-Score) |
| `sweep_topk_sensitivity.py` (alt) | 0.563 | 0.669 | **Erstes Sidecar-Token** (oft ``), nicht Action-Window; C1-Sidecars enthalten volle Thinking-Sequenz |
| H1a kanonisch (`−mean_entropy`, Episode `steps_detail.tle`) | **0.619** | **0.744** | Committed-action **Mittel** über Action-Tokens |

**Herkunft „0.19":** Nicht C-6 auf TW. `preanalysis_screen` auf `211029` meldet **ToH** raw TLE ≈ **0.209** (Flip ≈ 0.791); TW raw ≈ 0.714 (Flip ≈ 0.286). TW pre-DV `n_pos=8` macht beide Arms schwach interpretierbar.

**Fix (Code):**
- `signal_discrimination_report`: TLE-AUROC auf **−mean_entropy** (VC unverändert).
- `token_entropy.py`: `slice_action_logprob_tokens`, `mean_entropy_at_top_k`, `tle_mean_entropy_at_k_from_logprob_tokens`.
- `sweep_topk_sensitivity.py`: Action-Window-Slice + Mean-at-K + gleiche Labels wie H1a; C2-Sample per Referenz-`mean_entropy` wählen.

**Status:** C-6 **nicht abgeschlossen** bis Re-Sweep auf Pod bestätigt `|ΔAUROC| < ε` vs H1a. **K=20 unfreeze** bis dahin.

**N-Konsistenz (C-1 ↔ `105004`):** `execution.max_concurrent_episodes=32` in `signal_smoke.yaml`, `experiment_core.yaml`, Parity-Freeze (`run_metadata.json`: N=32, eps=0.05); `105004` `max_in_flight_observed=32`. **Identisch.**

## 2026-07-14 — C-5/C-6 complete; K=20 frozen

**Run:** `phase1_20260714_105004` — `signal_smoke.yaml`, 72 ep, quest-DV, cot_max_tokens=8192, git `3be18f0`, wall 47m 42s (~93 ep/h).

**C-5:** Pipeline + 72 logprob sidecars **Done**. TW `n_positive` (optimal_only) **81** vs `211029` **8** (pre-DV score labels).

**C-6 / K:** Siehe Rekonzilierungs-Eintrag oben — vorheriger K-Freeze **zurückgenommen** pending Re-Sweep.

**H1a-Smoke (kanonisch, `−mean_entropy`, optimal_only):** TW TLE 0.619 / VC 0.611; ToH TLE 0.744 / VC 0.701. **Nur Diagnose** (DV-Fix, Signal nicht tot); kein Design-Tuning.

**VC TW ≈ TLE TW:** Gleichstand auf 72 ep — empirische H1a-Frage, kein Bug; Fenster geschlossen.

## 2026-07-14 — C-5/C-6 pre-run declaration (post quest-DV)

**Vor `signal_smoke`-Re-Run festgehalten** — H1a-AUROC vor Freeze sichtbar; zulässige Entscheidungen:

| Erlaubt | Nicht erlaubt |
|---------|----------------|
| **K** wählen/einfrieren (C-6, beide Domänen, **dasselbe K**) | Schwellenwert-Tuning |
| Diagnose: TW AUROC nach DV-Fix **> 0.5** (Inversion weg?) vs `211029` tle≈0.19 | Signaldefinition ändern |
| Pipeline/Sidecar-Validierung | Collapse-Umschaltung |
| | Domänenentscheidungen / Phase-1 Go aus Smoke-AUROC |

**C-5 `211029`:** Sidecars/pipeline **Done**; TW H1a **nicht Done** (score-labels, n_pos=8, tle≈0.19 inverted). Re-Run @ `signal_smoke.yaml` 8192 + quest labels ersetzt Signal-Arm.

**Geplant:** 72 ep → AUROC TLE+VC × {optimal_only, legal_or_optimal} × domain + n_positive → C-6 K∈{5,10,20} → K freeze.

## 2026-07-14 — TextWorld DV repair: quest-distance labels (`9d994b8`)

**Zweck:** Abhängige Variable reparieren — TW-`optimal` von score-basiert auf quest-Restdistanz (`len(policy_commands)`) umstellen, analog ToH (strikt reduziert Restdistanz). **Begründung ausschließlich Konstruktvalidität:** (1) score-basiertes Label klassifizierte korrekte Navigationsschritte als `legal` → negative Klasse unter `y_optimal` kontaminiert (H1a); (2) TW und ToH maßen unter gleichem Label-Namen verschiedene Konstrukte (H4 DiD). **Nicht** begründet mit beobachteten Signal- oder AUROC-Werten.

**Reproduzierbarkeit:** Label-Re-Run **`phase1_20260714_100023`** lief auf Pod mit scp-Overlay vor Commit `9d994b8`. Pre-commit `ruff format` änderte danach nur Whitespace in `_score_progress_step` (scp MD5 `ef39ff27…` vs. Commit `9d994b8` MD5 `b37e9e9f…`; git diff = eine mehrzeilige Return-Expression, **keine Label-Logik**). **Kein Re-Run nötig** — semantisch identisch, akzeptiert. Pod @ `6eb0f5d` gepullt; Artefakt `100023` referenzierbar unter `9d994b8`/`6eb0f5d`.

| Bereich | Ergebnis |
|---------|----------|
| Commits | **`9d994b8`** (Implementierung) + **`6eb0f5d`** (Log-Anker) |
| Label-Regel TW | `optimal` iff ausführbar **und** `dist_after < dist_before` (strikt); `legal` iff ausführbar und dist ≥ vorher; `illegal` unverändert; `unlabeled` + `label_reason` wenn dist nicht berechenbar oder `policy_commands==[]` ohne `won` |
| Terminalschritt | `dist 1→0`, `won=True`, `policy=[]` → **optimal** (kein Unlösbar-Sonderfall) |
| Reason-Codes | `quest_distance_unavailable`, `quest_distance_empty_unwon` — **unit-getestet**, in `100023` **nicht ausgelöst** (0/90 TW-Steps); nicht als Feldbeobachtung führen |
| Score | `score_progress_step` (bool) als deskriptive Nebenvariable; nicht für `correctness` |
| EnvInfos | `policy_commands=True`, `intermediate_reward=True` |
| Unit-Tests | `tests/test_textworld_label.py` |
| Re-Run Probe | `phase1_20260714_100023` — `format_vc_probe`, cot_max_tokens=8192, max_steps=10 |

**Label-Verteilung (Re-Run, n=90 Steps/Domain, max 10 Steps/Episode):**

| Domain | optimal | legal | illegal | unlabeled | **n_positive** |
|--------|--------:|------:|--------:|----------:|---------------:|
| TextWorld (neu) | 37.8% | 34.4% | 27.8% | 0 | **34** |
| TextWorld (alt, score @ 083538) | 2.2% | 71.1% | 26.7% | — | **2** |
| ToH (unverändert) | 26.7% | 48.9% | 24.4% | 0 | **24** |

**FREEZE-REVIEW / §5.9 (Thesis-Repo, nicht Code):**
- Informationsbeschaffung (`look`/`inventory`/`examine`) distanzneutral → TW `legal`, in ToH absent; Observability-Asymmetrie (H4); `optimal_or_legal`-Sensitivität.
- **Rückkehr belohnt:** Label = Fortschritt vom aktuellen Zustand, nicht „auf globalem Optimalpfad“. Fehlzug (+1 dist) + Rückweg (−1 dist) → Rückweg `optimal`; Oszillation erzeugt alternierende Labelsequenz (ToH analog). `optimal`-Rate ≠ Lösungsqualität (Ergebnisteil); für H1a korrekt (Schrittkorrektheit relativ zum Zustand). **H3:** zeitliche Autokorrelation in Labels bei Oszillation — neben Solution-Space-Compression-Confound in §5.9 benennen; Inferenz via Cluster-Bootstrap/GEE abgefangen, Interpretationshinweis nötig.
- Episodenlänge TW: §5.5 8–15 vs. Probe 15–20+ → Phase-0-Kalibrierung; `position_norm` (H3).
- Instanz-Heterogenität: Reset-Restdistanzen 7/4/7.

**Nächster Schritt:** Gate C abgeschlossen (C-0…C-6). Phase-1-Readiness / Gate F ep/h aus `105004` (~93 ep/h @ 20 steps). C-1 freeze metadata **Done**.

## 2026-07-14 — Gate C-2/C-4 budget raster + C2 admissibility (`7b1ef9f`)

**Zweck:** Thinking-Budget einfrieren; C2-Vote-Pipeline korrigieren; format_vc_probe abschließen ohne H2-Metriken im Entscheidungspfad.

| Bereich | Ergebnis |
|---------|----------|
| Budget raster | 1024 → 2048 → 4096 → **8192** (TW+ToH, beide Thinking-Stufen); **frozen** in `experiment_core.yaml` |
| VC prompt | **OK** — `711785e`: trailing `Confidence:` entfernt; TW VC-Echo behoben |
| C2 admissibility | **OK** — `eac97cd`: Vote nur über closed+`post_think`; `truncation_no_action` + `n_samples_admissible` geloggt; verworfene Samples token-billed |
| `truncation_no_action` @8192 | **0/30 TW**, **0/29 ToH** — primäre post-fix C2-Metrik |
| C1 closure @8192 | **93.3% TW** (28/30), **100% ToH** — 2 TW-Ausfälle = **8192-token cap hits** (`textworld_2_C1` steps 8–9); Wilson ~78–98% @ n=30 → Pilot zeigt Budget nicht mehr limitierend; belastbare Rate in Phase 1 |
| avg `n_samples_admissible` | **2.87 TW**, **2.97 ToH** |
| VC | **179/179** (100%; 179 steps — one ToH C2 ep 9 steps) |
| `winner_closed` | **N/A as evidence** — by construction 100% nach Admissibility-Fix |
| Step labels (8192 run) | TW: optimal 2.2%, legal 71.1%, illegal 26.7%; ToH: optimal 34.8%, legal 40.4%, illegal 24.7% — ToH nicht degeneriert ~90%+ optimal |
| Artefakt | `phase1_20260714_083538` |

**Freeze-relevant (Thesis-Repo, nicht still):** §5.3 C2 = 3 Generierungen (voll billed), Vote über zulässige Kandidaten, effektives N protokolliert; Limitation §5.9.

**Nächster Schritt:** C-1 freeze metadata; C-6 topk; weiter Gate C — ToH-Schwierigkeit (3 vs 4 disks) vor Phase 1 prüfen falls TW/ToH label balance divergiert.

## 2026-07-13 — C-1 scoped waiver + N=32 freeze (design sign-off)

**Zweck:** C-1 Batch-Invarianz an den Signal-Kontrakt (committed-action-Fenster) angleichen und Produktions-N einfrieren.

| Bereich | Ergebnis |
|---------|----------|
| Entscheidung | **N=32 frozen** (explizite User-Freigabe); Gate auf committed-action-repräsentative Probes beschränkt |
| Committed-action Probes @ N=32 | **PASS** — `tw_short` 0.0002, `toh_short` 0.0078, `minimal_1` 0.0010, `minimal_2` 0.0021 (alle ≤ eps=0.05) |
| `minimal_3` | **Diagnostik** (non-gating): dtle_mean 0.09–0.15 durch Sequenz-Divergenz bei unterspez. Prompt; nicht ergebnis-relevant |
| `parity_prompts.json` | **OK** — `gating`/`role`/`note`; `minimal_3` → `gating:false` |
| `src/execution/parity.py` | **OK** — Gate über gating-Probes; `diagnostic_max_dtle` transparent berichtet; Default `gating=True` (rückwärtskompatibel) |
| Tests | **OK** — `test_non_gating_probe_drift_does_not_fail_gate`; Suite **281 passed** |
| Configs | **N=32** in `experiment_core.yaml` + allen `configs/dev/*.yaml` |
| Frozen params | `(N=32, eps=0.05 bits)`, `eps_derived_under_load=True` |

**Nächster Schritt (Pod):** Parity @ N=32 mit neuem Gate re-run (PASS + Freeze-Metadaten), dann `format_vc_probe` → `toh_parse_probe` → `signal_smoke`.

## 2026-07-13 — Gate C post-perf re-run (Pod online) — C-1 FAIL (superseded by scoped waiver above)

**Zweck:** Gate C nach `perf/vllm-server-concurrency` auf Pod fortsetzen; C-1 bei N=32 FAIL → N=24/N=16 Fallback.

| Bereich | Ergebnis |
|---------|----------|
| Throughput re-sweep | **OK** — N=8/16/24/32 smoke GO; best N=32 @ 192.4 ep/h (`throughput_sweep_post_perf.json`) |
| C-1 parity N=32 | **FAIL** — K-coverage PASS, temp PASS, batch FAIL (max_dtle=0.091 > eps=0.05) |
| C-1 parity N=24 | **FAIL** — batch max_dtle=0.154 |
| C-1 parity N=16 (fallback) | **FAIL** — batch max_dtle=0.133 |
| C-1 parity N=8 (diagnostic) | **FAIL** — batch max_dtle=0.130 |
| Worst case | Probe `minimal_3`, constellation `pool*_long` (96 filler tokens); domain probes (`tw_short`, `toh_short`) PASS |
| `experiment_core.yaml` | **N=16** (fallback; kein N erfüllt batch invariance) |
| C-2/C-4 format_vc_probe | **BLOCKED** — hard stop |
| C-3 toh_parse_probe | **BLOCKED** |
| C-5 signal_smoke | **BLOCKED** |

**Nächster Schritt:** vLLM batching/prefix-cache unter mixed-length concurrent load untersuchen; Parity erneut oder explizite Design-Freigabe für N=32 (max_dtle=0.091).


## 2026-07-13 — ServerBackend concurrency fix + Gate C re-run pending (Pod offline)

**Zweck:** Durchsatz-Engpass im HTTP-Client behoben; Gate-C-Läufe auf Pod müssen mit `perf/vllm-server-concurrency` wiederholt werden.

| Bereich | Ergebnis |
|---------|----------|
| `ServerBackend` | **OK** — globaler HTTP-Lock entfernt; C2 `n=` Batched Request; Timeout 600s |
| Testsuite lokal | **OK** — `python -m pytest tests/ -q` → **278 passed** |
| `docs/runpod.md` | **OK** — vLLM serve Tuning-Flags + aktualisierte Sweep-Kandidaten |
| `scripts/apply_production_n.py` | **OK** — N aus Sweep-JSON in YAML schreiben |
| `scripts/run_instrument_validation_after_perf.sh` | **OK** — Pod-Sequenz Sweep → Parity → Probes |
| C-1 Parity 2026-07-12 | **INVALIDIERT** — lief über serialisierten Client; **Re-run auf Pod** |
| Throughput / Gate C Pod | **ausstehend** — Pod `only_emerald_roundworm` offline; `bash scripts/run_instrument_validation_after_perf.sh` |

**Nächster Schritt (Pod):** Branch `perf/vllm-server-concurrency` pullen → vLLM mit neuen Flags → `run_instrument_validation_after_perf.sh` oder Schritte in `docs/runpod.md`.

## 2026-07-12 — Gate C Instrument-Validierung (erste Pod-Session, teilweise)

**Zweck:** Erste Live-Läufe auf RunPod 5090; C-0/C-3 teils PASS; C-2/C-4 blockiert; C-1 später invalidiert (siehe 2026-07-13).

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
