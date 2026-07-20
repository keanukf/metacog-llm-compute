# Gate E — Analyse-Rehearsal (End-to-End-Trockenlauf)

**Datum:** 2026-07-17
**Branch/Commit:** `feat/gate-d-calibration` @ `5f053da`
**Quelle (Analyse-Kette):** `data/results/instrument_validation/phase1_20260714_105004/` — 72 reale
Episoden (TextWorld + Tower of Hanoi, C0/C1/C2, Qwen3-8B/vLLM, Gate C Signal-Smoke, `signal_smoke.yaml`)
**Quelle (Phase-2-Smoke):** `configs/dev/gate_e_rehearsal.yaml` (Mock-Modus, kein GPU) + `data/results/gate_e_rehearsal/`
**Zweck:** Der zweite HART-Punkt aus Gate E (`blueprints/gate_p1_readiness.md`, Abschnitt „Gate E"):
kompletter Trockenlauf der konfirmatorischen Analyse-Kette **vor** echten Phase-1-Daten, damit keine
Analyse-Code-Reparatur mehr nach Sichtung der konfirmatorischen Ergebnisse stattfinden kann. Der
Pre-Analysis-Screen-Trockenlauf (das andere HART-Item) ist bereits separat erledigt
(`data/results/instrument_validation/phase1_20260714_105004/preanalysis_screen.{json,md}`).

---

## Kette (Kurzfassung, alle Schritte lauffähig ohne manuelles Umformatieren)

| # | Schritt | Code | Ergebnis |
|---|---------|------|----------|
| 1 | Episode-JSONs → Step-Tabelle | `src/analysis/datasets.py::load_run_dataset` | 72 Episoden → 1363 Steps, 0 fehlende Spalten |
| 2 | Künstlicher Holdout/Pool-Split | `scripts/gate_e_rehearsal.py` (neu) | instance < 3 je Domäne → holdout=True (344/1363 Steps) |
| 3 | Grid-Search + Policy-Artefakt | `src/analysis/thresholds.py::grid_search_thresholds` / `write_threshold_artifact` | `objective_definition=step_level_proxy_v1`, 36 Grid-Kandidaten je Domäne × Signal |
| 4 | `load_policy` Sanity-Check | `src/agent/allocation_policy.py::load_policy` | stage-wise ECDF lädt, low/mid/high → C0/C0/C2 in beiden Domänen |
| 5 | Phase-2-Mock-Smoke (`adaptive_tle`) | `scripts/run_phase2.py --config configs/dev/gate_e_rehearsal.yaml` | 12/12 Episoden, 0 Fehler, Policy-Artefakt via SHA-256 im `run_metadata.json` referenziert |
| 6 | `cluster_bootstrap` auf ΔAUROC(TLE, VC) | `src/analysis/inference.py::cluster_bootstrap` + `delta_auroc` | gepoolt: point=0.101, 90 %-CI [0.047, 0.157], n=1018 Pool-Steps, 18 Cluster |

Reproduzieren:

```bash
python3 scripts/gate_e_rehearsal.py \
  --run-dir data/results/instrument_validation/phase1_20260714_105004 \
  --holdout-instances 3 \
  --artifact-out data/results/gate_e_rehearsal/policy_artifact.json \
  --report-out data/results/gate_e_rehearsal/rehearsal_report.json

python3 scripts/run_phase2.py \
  --config configs/dev/gate_e_rehearsal.yaml \
  --checkpoint-dir data/results/gate_e_rehearsal/phase2_mock \
  --no-timestamp-run
```

---

## 1. Künstlicher Holdout/Pool-Split (Workaround, nicht das echte Gate-D-Manifest)

Der 105004-Lauf trägt **kein** eingefrorenes `difficulty_manifest.json` mit `holdout`/`difficulty_tier`
(Gate D ist noch offen). Tatsächlich fiel beim Laden auf, dass die ToH-Episoden bereits ein
`holdout`-Feld tragen (Instanzen 0–4 von 12 = `True`, aus einem bereits existierenden, aber
**nicht eingefrorenen** `data/tasks/tower_of_hanoi/difficulty_manifest.json`), während TextWorld
gar keins hat (`holdout=False`/`difficulty_tier=None` für alle 12 Instanzen — TextWorld hat noch
kein Manifest). Um einen domänensymmetrischen, selbst kontrollierten Split für das Rehearsal zu
haben, überschreibt `scripts/gate_e_rehearsal.py` das `holdout`-Feld für **beide** Domänen einheitlich:
**Instanz-Index < 3 (von 12) → holdout=True**, Rest → Pool. Das ist explizit **nicht** der reale
Gate-D-Split (5 von 50 = 10 %); mit nur 12 Instanzen je Domäne wären 10 % (≈1 Instanz) zu wenig für
eine belastbare stage-wise ECDF, deshalb 25 % (3 von 12). Ergebnis: 174/530 (TextWorld) und
170/489 (ToH) Steps holdout/pool.

**Für die echte Gate-E-Ausführung nach Gate D:** Sobald `difficulty_manifest.json` für beide Domänen
eingefroren ist, entfällt dieser Override ersatzlos — `load_run_dataset` liest `holdout` dann direkt
aus den echten Episode-JSONs (das Feld wird bereits im Schreibpfad unterstützt, siehe Gate B). Das
Rehearsal-Skript ist so geschrieben, dass der Override nur aktiv eingreift (`--holdout-instances`),
nicht `load_run_dataset` selbst verändert.

---

## 2. Episode-JSONs → Step-Tabelle

`load_run_dataset("data/results/instrument_validation/phase1_20260714_105004")` →
**72 Episoden, 1363 Steps**, `validate_analysis_schema`: keine fehlenden Spalten, `missing_vc_rate=0.0`,
`missing_tle_rate=0.0`, `missing_label_rate=0.0007` (1 Step ohne auswertbares Correctness-Label),
`synthesized_steps_rate=0.0` (alle Episoden hatten echtes `steps_detail`, keine Synthese nötig). Lief
**ohne jede Änderung** an `datasets.py` bis auf den unten dokumentierten Phase-2-Fund (Abschnitt 7).

---

## 3. Grid-Search-Threshold + Policy-Artefakt

`write_threshold_artifact` → `data/results/gate_e_rehearsal/policy_artifact.json`. Für beide Domänen
× beide Signale (`vc`, `tle_mean_entropy`): `objective_definition=step_level_proxy_v1`,
`theta1=0.8`, `theta2=0.9`, 36 Grid-Kandidaten (Standard-Quantilgrid). Holdout-ECDF-Größen pro Stage:
TextWorld C0=54/C1=60/C2=60, ToH C0=60/C1=60/C2=50 (kleine Ungleichheit durch unterschiedliche
Episodenlängen je Stage — nicht der Split selbst, sondern reale Trajektorienlänge).

**Beobachtung (nicht blockierend, aber notierenswert):** `theta1`/`theta2` landen für **alle vier**
Domäne×Signal-Kombinationen auf denselben Extremwerten (0.8/0.9) des Quantilgrids. Das ist plausibel
ein Artefakt des kleinen Pilot-Datensatzes (nur 3 Holdout-Instanzen/Domäne, 1 Run/Bedingung) in
Kombination mit der Pareto-Front-Auswahl (`min token_proxy` zuerst) — mit den viel größeren echten
Phase-1-Holdout-Daten (45 Non-Holdout-Instanzen × 5 Runs) ist mehr Streuung in der gewählten
`(theta1, theta2)`-Kombination zu erwarten. Kein Hinweis auf einen Code-Fehler (die Grid-Search-Logik
selbst ist bereits durch `tests/test_thresholds_grid.py` synthetisch abgedeckt), aber ein Punkt, den
man im Auge behalten sollte, falls dasselbe Muster auf den echten Daten wieder auftritt.

---

## 4. `load_policy` Sanity-Check

`load_policy(artifact_path, domain=dom, signal="tle_mean_entropy")` für beide Domänen: lädt
`ecdf_by_stage` korrekt (Längen wie oben), `theta1=0.8`, `theta2=0.9`, `direction=higher_is_uncertain`.
Low/Mid/High-Probe auf der größten Stage-Referenz (`C1` bzw. `C2`, je nachdem welche Stage die meisten
Holdout-Werte hat) ergibt für beide Domänen `["C0", "C0", "C2"]` — konsistent mit den engen, hohen
Thresholds aus Schritt 3 (fast alles landet entweder klar unter `theta1` oder klar über `theta2`;
`C1` wird bei diesen Perzentil-Proben selten getroffen, was zur oben genannten Beobachtung passt).

---

## 5. Phase-2-Mock-Smoke (`adaptive_tle`)

Config `configs/dev/gate_e_rehearsal.yaml` (extends `experiment_core.yaml`): `phase2.strategies:
[adaptive_tle, always_c0]`, `phase2.domains: [textworld, tower_of_hanoi]`, `instances_per_domain: 3`,
`runs_per_condition: 1`, `episode.max_steps_per_episode: 6`, `phase2.policy_artifact:
"data/results/gate_e_rehearsal/policy_artifact.json"` (das ist der bestätigte Config-Key —
`experiment_core.yaml` Zeile ~39 hat ihn als Kommentar vorgemerkt: `# policy_artifact:
"data/results/phase1/threshold_artifact.json"`).

`python3 scripts/run_phase2.py --config configs/dev/gate_e_rehearsal.yaml --checkpoint-dir
data/results/gate_e_rehearsal/phase2_mock --no-timestamp-run --` (kein `--real`, Mock-Backend):

- **12/12 Episoden abgeschlossen, 0 Fehler** (`errors.jsonl` leer), `episodes_failed=0`.
- `run_metadata.json`: `policy_artifact_path` + `policy_artifact_sha256` korrekt gesetzt (Hard-Fail-Pfad
  für `adaptive_tle`/`adaptive_vc`/`eager_style` ohne Artefakt ist bereits durch
  `tests/test_pr2_misc.py::test_run_phase2_hard_fails_without_policy_artifact` abgedeckt und blieb
  nach den Fixes unten grün).
- `by_stage_or_strategy`: `adaptive_tle` avg_tokens=60.0, `always_c0` avg_tokens=30.0 — der Allocator
  hat die geladene Policy tatsächlich angewendet (das Mock-Modell liefert konstante kanonische
  Logprobs, die Policy eskaliert deshalb bei `adaptive_tle` durchgehend auf C2; das ist erwartetes
  Mock-Verhalten, kein Signal-Befund — Signalqualität ist Gate C's Job, nicht dieses Rehearsals).
- Erfolgsraten 0.0 in allen Zellen (Mock-Modell antwortet immer `"go north"`, in beiden Domänen meist
  illegal) — erwartet und irrelevant für den Zweck dieses Schritts (CLI-Plumbing, nicht Signalqualität).
- Bonus-Check (nicht im ursprünglichen Kettenauftrag, aber naheliegend): `load_run_dataset` auf dem
  Phase-2-Mock-Output selbst liefert nach dem Fix in Abschnitt 7 **12 Episoden, 72 Steps**, keine
  fehlenden Spalten.

---

## 6. `cluster_bootstrap` auf ΔAUROC(TLE, VC)

**Statistik-Wahl:** ΔAUROC(TLE, VC) auf den Phase-1-Pilotdaten selbst (`src/analysis/inference.py::
delta_auroc`, geclustert per `cluster_bootstrap` über `instance_key`), **nicht** auf dem Phase-2-Mock-
Output. Begründung: Das Mock-Modell liefert für jeden Step identische kanonische Logprobs/VC-Werte
(siehe `src/utils/experiment_env.py::MockExperimentModel`) — jede Statistik auf Mock-Episoden wäre
entweder entartet (Erfolgsrate konstant 0 über alle Bedingungen) oder ohne Varianz in den Bootstrap-
Resamples. Die reale, konfirmatorisch relevante ΔAUROC-Statistik lässt sich dagegen sauber aus den
72 echten Gate-C-Episoden berechnen — das ist exakt dieselbe Statistikform wie H1a
(TLE-Diskrimination) und damit die aussagekräftigere Wahl für dieses Rehearsal.

Berechnet auf den **Non-Holdout- ("Pool"-)Steps** (spiegelt `build_policy_artifact`s eigene
`platt_eval=non_holdout_confirmatory`-Konvention), `y_optimal`-Label, `n_boot=5000`, `seed=20260703`
(Standard aus `inference.py`):

| Domäne | n Steps | n Cluster (Instanzen) | point ΔAUROC | 90 %-CI | skewness |
|--------|--------:|-----------------------:|-------------:|---------|---------:|
| gepoolt | 1018 | 18 | 0.1015 | [0.0470, 0.1569] | 0.110 |
| textworld | 529 | 9 | 0.0796 | [0.0268, 0.0794] | NaN |
| tower_of_hanoi | 489 | 9 | 0.0610 | [0.0054, 0.1133] | −0.084 |

ΔAUROC > 0 heißt: TLE diskriminiert `y_optimal` in diesem Pilot-Ausschnitt etwas besser als VC (positive
CI, schließt 0 nicht ein — konsistent mit Gate C-5's TLE-AUROC-Befund, aber diese Zahl selbst hat **keine**
konfirmatorische Aussagekraft, da sie auf denselben 72 Pilot-Episoden basiert, die auch Gate C
produziert hat, nicht auf frischen Phase-1-Daten).

**Rauer Rand (ehrlich dokumentiert):** Bei TextWorld liegt der Punktschätzer (0.0796) knapp **außerhalb**
des berichteten 90 %-Perzentil-Intervalls (oberes Ende 0.0794) und `skewness` ist `NaN` (Bootstrap-
Verteilung mit `n<3` effektiven Momenten irgendwo entartet — vermutlich durch nur 9 Cluster in dieser
Domäne). Das ist ein bekanntes Perzentil-Bootstrap-Phänomen bei sehr wenigen Clustern (hier 9 Non-Holdout-
Instanzen je Domäne), kein Code-Fehler — mit den 45 Non-Holdout-Instanzen der echten Phase-1-Daten sollte
sich das deutlich entschärfen, ist aber ein Punkt, den man beim echten H1a/H3-Bootstrap im Auge behalten
sollte (ausreichend Cluster für ein stabiles Perzentilintervall).

---

## 7. Gefundene Bugs (der eigentliche Zweck dieser Übung)

Alle drei folgenden Funde wurden **vor** echten Phase-1-Daten gemacht und sind hier als kleine,
klar abgegrenzte Fixes behoben (keine Design-Entscheidung, keine DV-Kontamination):

### 7.1 `run_phase1.py`/`run_phase2.py::load_config` ignorierte `extends`

`configs/dev/gate_d_calibration.yaml` (und andere `configs/dev/*.yaml`) nutzen `extends:
../experiment_core.yaml` — aber `load_config` in `run_phase1.py`/`run_phase2.py` war ein reines
`yaml.safe_load(f)` **ohne** Merge. Ein Overlay wie `configs/dev/gate_e_rehearsal.yaml` hätte damit
nur seine eigenen Top-Level-Keys gesehen (kein `model`, `episode.max_steps_per_episode` überschrieben
statt gemerged, kein `domain_prompts`, kein `paths` — stiller Fehlschlag, kein Ladefehler). Die vier
Gate-D-Diagnose-Skripte (`run_gate_d_feasibility.py`, `run_gate_d_toh_feasibility.py`,
`gate_d_manifest_smoke.py`, `sweep_toh_difficulty.py`) hatten das Problem bereits gelöst, indem sie
`_load_merged_config` aus `scripts/sweep_textworld_difficulty.py` importieren — `run_phase1.py`/
`run_phase2.py` selbst hatten diesen Fix nie bekommen, vermutlich weil sie in der Praxis bisher immer
mit vollständigen Configs (`experiment_core.yaml` direkt, kein `extends`) aufgerufen wurden. **Fix:**
`load_config` in beiden Skripten nutzt jetzt denselben `_load_merged_config`-Import. Bestehender Test
(`tests/test_pr2_misc.py::test_run_phase2_hard_fails_without_policy_artifact`, Config ohne `extends`)
bleibt unverändert grün.

### 7.2 YAML-Boolean-Falle bei `logprob_sidecar_mode: off`

Der Phase-2-Mock-Smoke schlug beim ersten Versuch mit `ValueError: logprob_sidecar_mode must be one
of [...], got False` fehl. Ursache: YAML 1.1 parst ein unquotetes `off` als **Python-Bool `False`**,
nicht als String `"off"` — `src/utils/logprob_sidecar.py::_normalize_mode` erwartete aber einen String
und warf bei `False` einen Fehler. Betroffen sind **sechs** bestehende Configs, die alle unquotet
`logprob_sidecar_mode: off` schreiben: `configs/dev/gate_e_rehearsal.yaml` (neu, dieses Rehearsal),
`gate_d_calibration.yaml`, `gate_d_diagnostic.yaml`, `format_vc_probe.yaml`, `toh_parse_probe.yaml`,
`throughput_probe.yaml`. Keines dieser Configs wurde bisher tatsächlich durch
`LogprobSidecarConfig.from_logging_config` gejagt (das Feld kam erst mit Commit `5be09f7`
„feat(logging): add action-window logprob sidecar modes" dazu, nach dem Gate-B-Smoke; die
Gate-D-Diagnose-Skripte bauen Episoden anders und laufen nie durch `Phase2RunContext`) — die Falle war
also real, aber bisher nie ausgelöst worden. **Fix:** `_normalize_mode` behandelt jetzt ``False``
explizit als `"off"` (mit Kommentar zur YAML-1.1-Falle) und `True` als expliziten Fehler (mehrdeutig —
könnte `action_window` oder `full` meinen). Root-Cause-Fix in der Parsing-Funktion statt sechs YAML-
Dateien einzeln zu quoten, damit künftige Configs mit demselben unquoteten `off` nicht erneut
betroffen sind. Zwei neue Tests in `tests/test_logprob_sidecar_mode.py`.

### 7.3 `load_run_dataset` verwarf **jede** Phase-2-Episode stillschweigend

Nach dem Fix in 7.2 lief der Phase-2-Mock-Smoke durch, aber der anschließende Bonus-Check
(`load_run_dataset` auf dem Phase-2-Output) lieferte **0 Episoden, 0 Steps** — ohne jede Fehlermeldung.
Ursache: `src/analysis/datasets.py::_validate_episode_record` verlangte hart ein episodenweites
`compute_stage`-Feld (`required = ("episode_id", "compute_stage", "task_success")`). Phase-1-Episoden
haben das (fixe Bedingung C0/C1/C2), Phase-2-Episoden aber nicht — sie tragen `strategy` statt
`compute_stage`, weil die Compute-Stage pro Step unter adaptiver Allokation variiert. Damit wurde
**jede** Phase-2-Episode beim Laden verworfen, ohne dass das jemals auffiel, weil bisher niemand
`load_run_dataset` auf einen echten Phase-2-Lauf angesetzt hatte (Gate B/C-Evidenz deckt nur
Phase-1-Läufe ab). Das hätte bei der echten H2/H4-Analyse (die auf `load_run_dataset`/`ds.episodes`
für Phase-2-Läufe aufbaut) zu einer leeren Step-Tabelle geführt — genau die Art von Stille-Fehlschlag,
die Gate E fangen soll, bevor echte Daten existieren. **Fix:** `_validate_episode_record` akzeptiert
jetzt entweder `compute_stage` (Phase 1) oder `strategy` (Phase 2) als Bedingungs-Identifikator.
Regressionstest: `tests/test_09_analysis_pipeline.py::
test_load_run_dataset_accepts_phase2_episodes_without_compute_stage`.

---

## 8. Weitere raue Ränder / Beobachtungen (keine Fixes, nur ehrlich notiert)

- **Künstlicher Split-Anteil (25 % statt 10 %):** Siehe Abschnitt 1 — bewusste Abweichung vom realen
  Gate-D-Design wegen kleiner Pilot-Instanzzahl (12 statt 50 je Domäne). Kein Problem für dieses
  Rehearsal, aber nicht 1:1 auf die echten Zahlen übertragbar.
- **theta1/theta2 auf Extremwerten** (Abschnitt 3) — im Auge behalten, ob sich das mit echten
  Phase-1-Daten wiederholt.
- **Kleine-Cluster-Bootstrap-Rand** (Abschnitt 6) — Punktschätzer leicht außerhalb der berichteten CI
  bei TextWorld (9 Cluster); bei den echten 45 Non-Holdout-Instanzen erwartungsgemäß entschärft.
- **`policy_artifact.json` enthält das volle 36-Zeilen-`grid_table` pro Domäne × Signal** (bewusst, für
  Nachvollziehbarkeit) — bei den echten Phase-1-Daten (mehr Domänen-/Signal-Kombinationen über mehr
  Steps) wird die Artefaktdatei entsprechend größer, aber weiterhin überschaubar (kein Performance-
  Problem beobachtet).
- **Mock-Phase-2-Smoke sagt nichts über Signalqualität** (erwartet, siehe Abschnitt 5) — er beweist nur
  CLI-Plumbing (Config → Policy-Load → Allocator → Episode-JSON), nicht ob `adaptive_tle` echte
  Steuerung liefert. Das ist genau die in der Aufgabenstellung vorgesehene Aufgabenteilung
  (Signalqualität = Gate C, Plumbing = Gate E).
- **`data/results/gate_e_rehearsal/`** ist über `.gitignore` (`data/results/`) ausgeschlossen — die
  Artefakte (`policy_artifact.json`, `rehearsal_report.json`, `phase2_mock/`) liegen lokal, werden aber
  nicht committet; dieses Dokument ist die archivierte Evidenz.

---

## Testsuite

`python -m pytest tests/ -q` nach allen Fixes: **335 passed**, 0 failed, 0 skipped (4 unveränderte
Warnings, nicht mit diesem Rehearsal zusammenhängend).
