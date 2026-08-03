# Consistency log

Durchlaufendes Verifikationslog für Thesis–Code-Abgleich. Neue Einträge mit Datum oben anfügen.

## 2026-08-03 — Policy: etablierte Libraries statt Eigenimplementierung; H1a-Holdout-Frage geklärt

**Policy-Entscheidung (Nutzer):** Wo immer eine etablierte, zitierfähige Library-Implementierung
zu unseren Daten passt, wird sie bevorzugt gegenüber einer eigenen Implementierung — leichter zu
verteidigen und zu evaluieren als eine Prüfung unserer eigenen Formel-/Tie-Break-Logik. Betrifft
konkret:

- **`compute_auroc`** (`src/analysis/calibration.py`): von einer handgeschriebenen
  Mann-Whitney-U-Rangsummen-Implementierung auf `sklearn.metrics.roc_auc_score` umgestellt. Vorher
  an 200 Zufallsstichproben (inkl. Ties) gegen sklearn verifiziert: max. Abweichung 1.1e-16
  (Maschinenpräzision) — die alte Implementierung war korrekt, die neue ist es auch, jetzt aber
  gegen eine zitierfähige Quelle statt gegen unsere eigene Rangberechnung zu verteidigen. Zitat für
  die Prosa: AUROC = normalisierte Mann-Whitney-U-Statistik (Hanley & McNeil, 1982) — bereits so in
  `chapters/05_methodology.md` §5.8 zitiert.
- **`holm`/`bh`** (`src/analysis/inference.py`): von einer handgeschriebenen Step-down/Step-up-
  Implementierung auf `statsmodels.stats.multitest.multipletests(method="holm"/"fdr_bh")`
  umgestellt. War vorher schon gegen genau diese Funktion cross-verifiziert (siehe Docstring-
  Historie); jetzt direkt darauf delegiert statt nur dagegen zu testen. Zitate: Holm (1979) für die
  Family-A–E-Korrektur, Benjamini & Hochberg (1995) für die explorative FDR-Schicht — beide bereits
  in `chapters/05_methodology.md` §5.8 zitiert.
- **`_skewness`** (`src/analysis/inference.py` und `src/analysis/descriptive_stats.py`): auf
  `scipy.stats.skew(arr, bias=True)` umgestellt (an 100 Zufallsstichproben gegen die alte Formel
  verifiziert: max. Abweichung 5.6e-16). Dabei einen echten kleinen Robustheitsgewinn mitgenommen:
  die Degenerationsprüfung prüft jetzt `np.allclose(arr, arr[0])` ("praktisch konstant") statt
  exaktem `std == 0` — scipy warnte bei einem quasi-konstanten Bootstrap-Replikat-Array (Testfall
  `test_h2_paired_holds_when_ci_bound_clears_threshold`, exakt null Instanz-zu-Instanz-Varianz) vor
  "catastrophic cancellation"; die alte Formel hätte an derselben Stelle still einen numerisch
  bedeutungslosen Wert geliefert, ohne zu warnen.
- **`_quantile`** (`src/analysis/preanalysis_screen.py`): auf `numpy.percentile` umgestellt. Der
  Kommentar "dependency-free by design" war ohnehin schon veraltet — das Modul importiert
  `cluster_bootstrap`/`estimate_icc`, die selbst längst numpy/scipy/statsmodels ziehen.
- **`requirements.txt`/`pyproject.toml`**: `statsmodels` und `scikit-learn` waren als feste,
  unbedingte Imports in Kern-Analysecode (`fit_h3_model`, `gee_icc`, `fit_tle_calibrator`, jetzt
  auch `compute_auroc`) im Einsatz, aber nur in `pyproject.toml`s optionalem `analysis`-Extra
  deklariert, nicht in `requirements.txt`. Jetzt in beide als harte Kern-Dependencies verschoben.

**Bewusst NICHT umgestellt — ICC (`src/analysis/icc.py`):** `pingouin` steht zwar schon als
optionale Dependency bereit, passt aber inhaltlich nicht auf unseren Anwendungsfall. Pingouins
`intraclass_corr` ist für klassische Rater-Reliabilitätsstudien gebaut (K Rater bewerten dieselben
N Zielobjekte je einmal) — wir haben aber keine "Rater", sondern wiederholte binäre Beobachtungen
(Steps) pro Instanz mit stark unterschiedlicher Anzahl je Instanz. Eine künstliche "Rater"-Zuordnung
(z.B. Step-Index innerhalb der Instanz) wäre semantisch bedeutungslos und würde eine
Reliabilitäts-Frage vortäuschen, die wir gar nicht stellen — wir fragen nach der
Varianzkomponente durch Clustering, nicht nach Bewerter-Übereinstimmung. `gee_icc` nutzt bereits
`statsmodels.GEE` (Library); `anova_icc1` bleibt die handgeschriebene Shrout-&-Fleiss-ICC(1)-
Formel als unabhängiger Cross-Check, weiterhin gegen `gee_icc` auf echten Daten verifiziert (siehe
Preanalysis-Screen-Report, beide Domänen stimmen überein).

**H1a-Holdout-Frage geklärt (Nutzerfrage):** H1a läuft auf allen 50 Instanzen pro Domäne, nicht nur
den 45 Nicht-Holdout-Instanzen — das ist beabsichtigt, nicht übersehen. Direkt in
`chapters/05_methodology.md` §5.8 verifiziert: der H1a-Absatz ("$Q_{\mathrm{disc}}$ ... computed
per domain ...") enthält keine Holdout-Einschränkung, im Unterschied zu den Absätzen für die
Schwellenwert-Suche ("estimated on the Phase 1 holdout") und H1b ("fitted on the holdout
instances"). Der Grund: H1a fitted nichts — AUROC ist eine rangbasierte, parameterfreie Messung
direkt auf dem (vorzeichenkorrigierten) Rohsignal. Es gibt keine Zirkularitätsgefahr wie bei
Schwellenwertsuche/Kalibrator (dort wird etwas auf Daten gefittet, das dann evaluiert werden
soll) — deshalb keine Notwendigkeit, Holdout auszuschließen. `stage2_h1a_discrimination.py` filtert
entsprechend nicht nach `holdout`, geprüft direkt im Code.

**Volle Suite:** 453 Tests grün nach dem Swap; komplette Pipeline (`run_all.py`) einmal gegen die
echten Daten neu durchlaufen, alle Stage-Outputs zahlenmäßig identisch zum Stand vor dem Swap
(reiner Implementierungswechsel, keine Verhaltensänderung — siehe Diff-Verifikation im gleichen
Arbeitsschritt).

## 2026-07-28 — Kanonischer Phase-1-Datensatz erstmals im Code festgeschrieben

Die realen Phase-1-Daten liegen in zwei Verzeichnissen, weil `phase1_20260722_091125`'s
TextWorld-Hälfte durch den bereits bekannten Stub-Environment-Bug (Commit `b47e35d`) verunreinigt
ist (45/50 Instanzen unwinnable, 5.3% Erfolgsrate) und separat in
`textworld_regen_20260724` (750 valide Episoden, 53.7% Erfolgsrate) neu gesammelt wurde. Diese
Auswahlregel war bisher **nirgends im Repo dokumentiert**, auch nicht hier im Log — nur in der
Commit-Message von `b47e35d` selbst.

**Jetzt erstmals im Code festgeschrieben:** `src/analysis/phase1_canonical.py` — kanonischer
Datensatz = `tower_of_hanoi` aus `phase1_20260722_091125` + `textworld` aus
`textworld_regen_20260724`, 1500 Episoden gesamt (750+750). `assert_canonical_invariants()` prüft
das hart (Gesamtzahl, pro Domain, pro Domain×Stage, Holdout-Instanzen pro Domain) und bricht laut
ab bei jeder Abweichung — direkt gegen die echten Daten verifiziert: 1500 Episoden, alle
Invarianten grün. `scripts/phase1_analysis/stage0_build_canonical_dataset.py` schreibt daraus ein
Manifest (`content_hash` über die sortierte Episode-ID-Liste) statt die ~36GB Rohdaten zu
kopieren; zweimal hintereinander ausgeführt liefert denselben `content_hash` (Idempotenz direkt
verifiziert). Erster Baustein der Phase-1-Analyse-Pipeline (`feat/phase1-real-analysis`).

## 2026-07-28 — P0-7 (H2 CI-Grenze statt Punktschätzer) und P1-stat-7 (Prompt-Token-Tracking)

**P0-7:** `h2_paired()` (`src/analysis/inference.py`) entschied bisher über den rohen
Punktschätzer der gepaarten Mittelwerte, nie über eine Bootstrap-CI-Grenze — obwohl sowohl
`notes/praeregistrierung_auswertungsplan.md` als auch die aktuelle Ch.5-Prosa explizit die
CI-Grenze verlangen. Per Grep bestätigt: die Funktion wurde nirgends im Code mit
`cluster_bootstrap` komponiert. Jetzt gefixt — beide Statistiken werden über Instanzen
bootstrap-resampled, Entscheidung läuft über die untere CI-Grenze. Volle Argumentation:
`docs/adrs.md` ADR-007. Regressionstest konstruiert einen Fall mit Mittelwert 0.0 (Punktschätzer
hätte "hält" gesagt) aber hoher Varianz, wo die CI-Grenze korrekt "hält nicht" sagt.

**P1-stat-7:** Prompt-/Input-Token-Tracking eingebaut (`total_prompt_tokens` pro Episode,
`prompt_tokens` pro Step), gemäß Nutzer-Entscheidung vom 2026-07-28 (Option a) — vor Run 2, nicht
rückwirkend für Phase 1 (Ökonomie-Entscheidung, keine Phase-1-Analyse braucht das Feld). Kommt aus
`usage.prompt_tokens` in der vLLM-Serverantwort, die bisher gelesen und dann verworfen wurde.
Rückwärtskompatibel über `GenerateResult` (verhält sich exakt wie ein 2-Tupel für die ~45
bestehenden Test-Mocks und alle Call-Sites, trägt zusätzlich `.prompt_tokens`). Ein subtiler
Fund unterwegs: `src/utils/logging_utils.py::_MINIMAL_STEP_KEYS` hätte das neue Feld beim
Compact-Storage-Schritt (Produktions-Default) still wieder rausgefiltert — gefangen durch einen
dedizierten Regressionstest, nicht zufällig entdeckt. Volle Argumentation: `docs/adrs.md` ADR-008.
Volle Suite grün (387 Tests, davon 6 vorbestehende an die neue, rückwärtskompatible Arity
angepasst).

## 2026-07-27 — P0-5 (Revision-Audit): H3-Signal-Standardisierung von Mean-Centering auf
stage-weises Z-Standardisieren korrigiert

**Anlass:** `../metacog-thesis/notes/revision_audit_2026-07.md` (Cross-Check Prosa ↔ frozen Code
vor Phase-1-Analyse), Punkt P0-5. Ch.5 §5.2.1/§5.8 behaupten, das Signal gehe "z-standardisiert" in
die H3-Modelle ein; `fit_h3_model` hat tatsächlich nur zentriert (`z - z.mean()`), nie durch die SD
geteilt, und dabei über alle drei Compute-Stages (C0/C1/C2) gepoolt statt stage-weise.

**Vor der Änderung real geprüft (nicht nur behauptet):** ToH-Manifest, TextWorld-Manifest und
`experiment_core.yaml` direkt gegen die Checklisten-Annahmen gegengecheckt (4 Disks alle 50
Instanzen, `random_scramble`, C0/C1/C2-Temperaturen, K=20, Holdout mod-10 je Domain) — zusätzlich
die ToH-4-Disk-Annahme nicht nur im Manifest, sondern in den tatsächlichen Prompt-Texten von 3
realen Episoden aus dem echten Run verifiziert (Instanzen 0, 25, 49: "Goal state: Peg C holds all 4
disks", keine Disk 5). Alle Checklisten-Annahmen bestätigt.

**Entscheidung + volle Argumentation:** `docs/adrs.md` ADR-006. Kurzfassung: stage-weise
z-standardisieren (nicht nur domain-weit zentrieren), weil TLE/VC über C0/C1/C2 wegen
unterschiedlicher Decoding-Temperaturen und Reasoning-Token-Budgets nicht direkt vergleichbar sind
— konsistent mit der stage-weisen ECDF-Normalisierung, die der Phase-2-Allocator bereits nutzt.

**Umsetzung:** `src/analysis/inference.py::fit_h3_model` gruppiert jetzt per `compute_stage` vor
der Standardisierung; ein Stage mit Varianz null/undefiniert lässt den Fit jetzt laut fehlschlagen
statt einen degenerierten Koeffizienten still zu produzieren. Zwei neue Regressionstests in
`tests/analysis/test_inference.py`. Volle Suite grün (374 Tests).

## 2026-07-21 — Struktureller Repo-Refactor: Skripte/Tests umsortiert, toter Code entfernt

**Zweck:** Rein struktureller Housekeeping-Refactor (Gate A–F alle abgeschlossen), mit harter
Zero-Functional-Change-Vorgabe. Über eine Multi-Modell-Pipeline ausgeführt: Oper plant ein
eingefrorenes Move/Rename-Manifest + Do-not-touch-Liste, ein Ausführungsmodell macht nur die
strukturellen Moves, ein unabhängiges Modell reviewt den Diff, ein separater code-frozen
Doku-Pass zieht Kommentare/Docs nach.

**Was passiert ist:** 55 Skripte + 12 Testdateien in zweckbenannte Unterordner verschoben/umbenannt
(`scripts/experiment/`, `scripts/datasets/`, `scripts/difficulty_calibration/`,
`scripts/instrument_validation/`, `scripts/analysis_rehearsal/`, `scripts/run_readiness/`,
`scripts/pilot_analysis/`, `scripts/cloud/{shell,python}/`); Gate-A–F-Jargon aus Datei- und
Testnamen entfernt (z. B. `gate_d_metrics.py` → `difficulty_metrics.py`,
`gate_f_c1c2_quality_probe.py` → `c1_c2_quality_probe.py`); numerische Präfixe der neun
`test_0N_*.py`-Dateien gestrichen. Zusätzlich 11 bestätigt tote Dateien entfernt: die tote
Prompt-A/B-Werkzeugkette (`run_prompt_ab.py` + `configs/prompt_variants/`), die tote
Multi-Modell-Vergleichs-Infrastruktur (`run_pilot_models.py`, `summarize_pilot_batch.py`,
`configs/models.yaml`, `configs/models_runpod.yaml`), ein ungenutzter Signal-Stub
(`src/signals/semantic_consistency.py`) plus `configs/experiment_ext.yaml`, ein redundanter
Validierungs-Wrapper (`run_tle_invariance_validation.py`) und ein Einmal-Wartungsskript
(`restore_cursor_plans.sh`).

**Verifikation:** Null Funktionsänderungen — bestätigt durch ein unabhängiges Diff-Review plus die
komplett grüne Testsuite nach jedem Batch (367 Tests). Der code-frozen Doku-Pass hat außerdem eine
vorab freigegebene tote `sys.path`-Insert-Zeile in `benchmark_inference.py` entfernt (nachweislich
inert).

**Historische Einträge:** Sämtliche Dateinamen in *früheren* Einträgen dieses Logs beziehen sich auf
die Pre-Refactor-Pfade (z. B. `gate_e_rehearsal.py`, `gate_f_c1c2_quality_probe.py`) und bleiben
bewusst unverändert — Append-only-Prinzip, gleiches Vorgehen wie bei `docs/adrs.md` ADR-001
(stale-but-labeled historischer Datensatz).

**Testsuite:** unverändert grün (367), keine inhaltliche Quelländerung außer der einen freigegebenen
Dead-Code-Entfernung.

## 2026-07-21 — Freeze-Tag-Timing final geklärt: erst nach Refactor + Re-Verifikation

**Zweck:** User fragte nach dem letzten offenen Gate-D-Punkt (Manifest-Freeze-Tag). Inhaltlich ist
nichts mehr offen (beide Manifeste final, committed, Mock-Smoke bestanden) — nur der eigentliche
git tag fehlt noch. Da Gate F seit heute komplett abgeschlossen ist, wäre die im Dokument selbst
genannte Vorbedingung ("Tag nach Gate F") jetzt technisch erfüllt.

**Entscheidung (User, per Rückfrage bestätigt):** Tag trotzdem nicht jetzt setzen, sondern erst
nach dem geplanten Repo-Refactor und der finalen Re-Verifikation aller eingefrorenen Configs.
Begründung: Der Tag soll den tatsächlichen Endzustand markieren, mit dem Phase 1 real läuft — nicht
einen Zwischenstand, der durch die anschließende Refactor-Arbeit sofort wieder überholt würde
(auch wenn der Refactor rein strukturell ist und keine Funktion ändern soll, wäre der getaggte
Commit sonst nicht der, der tatsächlich für Phase 1 verwendet wird).

**Testsuite:** keine Quelländerung (nur Dokumentation).

## 2026-07-21 — Gate F komplett abgeschlossen: C1/C2-Fix real bestätigt, Run-Hygiene 3× bestätigt, neues Progress-Tool

**Zweck:** PR #25 (C1/C2-Vereinheitlichung) gemergt; realer Recheck auf neu gestartetem Pod
(`e21lf03kz5htos`, via `runpodctl` neu hochgefahren) nachgeholt, da der Mock-Backend den
`</think>`-Bug-Fall nicht realistisch simulieren kann.

**Neuer Lernpunkt zum Pod-Handling:** `runpodctl stop` räumt die Container-Disk komplett leer (nur
`/workspace` bleibt persistent) — jeder Neustart braucht ein volles `setup_cloud.sh` (venv + Modell
neu, ~einige Minuten), nicht nur ein Reconnect. Zusätzlich ändert sich der öffentliche SSH-Port bei
jedem Neustart (zweimal bestätigt, gleiche Pod-ID, unterschiedlicher Port) — `~/.ssh/config` muss
danach jedes Mal aktualisiert werden (`runpodctl get pod <id> -a`, Spalte `PORTS`).

**Run-Hygiene:** dritte unabhängige Bestätigung, wieder 5/5. Checkbox abgehakt.

**C1/C2-QC-Recheck (n=2/Zelle, 8 Episoden real, N=16 parallel, Laufzeit ~30 Min statt der ~59 Min
gestern bei n=5):**

| Zelle | Ergebnis |
|---|---|
| TextWorld/C1 | Kein `<think>`-Leak mehr — "no action parsed at all" statt literalem String. 3 legitime `cot_max_tokens=8192`-Trunkierungen. |
| TextWorld/C2 | 6 legitime `thinking_unclosed`-Rejections (derselbe, schon vorher korrekte Mechanismus). |
| ToH/C1 | Sauber, keine Befunde. |
| ToH/C2 | Sauber, keine Befunde. |

Voting-Korrektheit erneut unabhängig bestätigt (kein einziger Fall, in dem der aufgezeichnete
Sieger vom neu berechneten Mehrheitsvotum abwich). **Gate-F-C1/C2-QC-Checkbox abgehakt.**

**Damit ist Gate F komplett abgeschlossen** (alle HART-Punkte + WEICH-Punkt gestrichen/erledigt).
Go/No-Go-Tabelle aktualisiert.

**Datensauberkeit:** Lokaler Sync hatte kurzzeitig alte (Pre-Fix, n=5) und neue (Post-Fix, n=2)
Episodendaten gemischt in `data/results/gate_f_c1c2_quality_probe/` (rsync ohne `--delete` lässt
Dateien liegen, die remote nicht mehr existieren). Mit `rsync --delete` bereinigt, lokal jetzt exakt
deckungsgleich mit dem Pod-Stand (25/25 Dateien).

**Neues Tool: `scripts/progress_watch.py`.** Auf Nutzerwunsch — leichtgewichtiger, von den
eigentlichen Run-Scripts komplett entkoppelter Fortschritts-Beobachter. Schaut nur von außen auf
ein Output-Verzeichnis (kein Hook in `run_phase1.py`/`run_phase2.py`/Probe-Scripts nötig), zählt
fertige Episoden (`ep_*.json`/`qc_*.json`) und laufende (`trace_*.jsonl`-Zeilenzahl), gruppiert nach
Domain/Stage per Dateinamen-Regex, meldet grobe Steps/Min-Rate. Live gegen den laufenden Pod-Recheck
getestet (erkannte korrekt 8/8 fertig, keine offenen Traces mehr). 6 neue Tests
(`tests/test_progress_watch.py`).

**Testsuite:** 367 passed (361 + 6 neue Progress-Watch-Tests).

## 2026-07-21 — C1/C2-Reasoning-Engine vereinheitlicht (fixt den TextWorld-C1-`<think>`-Fund)

**Zweck:** Der C1-`<think>`-Leak-Fund vom 2026-07-20 (`docs/consistency_log.md`, Eintrag "C1/C2-
Qualitätskontroll-Probe gelaufen") gefixt — auf Nutzerwunsch nicht als isolierter Patch, sondern
durch eine echte Vereinheitlichung: C1 ist jetzt strukturell "C2 mit n_samples=1, kein Voting"
statt einer separaten, unabhängig gedrifteten Implementierung.

**Root Cause bestätigt:** `src/agent/cot_parser.py::parse_cot_action()` ist laut eigenem Docstring
"Structured parse result for C1 CoT outputs" — für C1 gebaut, aber nie von C1 genutzt. C1 rief
stattdessen das naive `_normalize_action_line()` auf (nur *komplette* `<think>...</think>`-Paare
werden entfernt; bleibt der Block offen, wird die erste Zeile — `<think>` — als Aktion
durchgereicht). C2 hatte den korrekten Mechanismus (`assess_c2_sample_admissibility`,
`thinking_unclosed`-Rejection) schon, nur nicht geteilt.

**Zweiter Fund beim Nachschauen:** C1 und C2 nutzten unterschiedlichen Prompt-Text. C1 instruiert
explizit über `<think>`-Tags; C2 nutzte denselben knappen `_SINGLE_LINE_OUTPUT_INSTRUCTION`-Text
wie C0 (das *kein* Thinking hat) und verließ sich rein auf den Generation-Parameter
`enable_thinking=True`. Kein inhaltlicher Grund für den Unterschied erkennbar — sieht nach
organischem Drift aus (C2 vermutlich aus C0s Code-Pfad abgeleitet, Instruktion nicht angepasst).
Auf Nutzerentscheidung vereinheitlicht: beide nutzen jetzt C1s ursprünglichen, bereits kalibrierten
Wortlaut (`shared._REASONING_OUTPUT_INSTRUCTION`).

**Auswirkung auf bestehende Ergebnisse:** Keine der Gate-D-Korridorentscheidungen ist betroffen —
TextWorld wurde gegen C0 kalibriert, ToH gegen C1; C2 war nie Referenzstufe in beiden Domänen. Der
bereits dokumentierte "C1>C2"-Befund (Basis für die "3 Compute-Stages bleiben"-Entscheidung) wurde
mit C2s altem Prompt gemessen und ist damit technisch überholt — die zugrundeliegende Erklärung
(Tie-Break-RNG, Self-Consistency-Theorie) bleibt unabhängig vom Prompt-Wortlaut gültig. Erneute
Messung geplant im Rahmen der ohnehin vorgesehenen Nachtests nach dem Repo-Refactor.

**Umsetzung:** Neue Funktion `shared.reasoning_step_core()` — die eigentliche Engine (N Kandidaten
generieren, `</think>`-Admissibility-Check, `parse_cot_action()`, TLE *nur* bei zulässigem
Kandidaten, Mehrheitsvotum). `c1_step_core`/`c2_step_core` sind jetzt dünne Wrapper, die dieselbe
Engine mit `n_samples=1` bzw. `n_samples=3` aufrufen — externe Config-Parameter (`c1_cot_temperature`,
`c2_cot_max_tokens`, etc.) unverändert, keine Breaking Changes an `configs/experiment_core.yaml`.
`majority_vote()` von `c2.py` nach `shared.py` verschoben (zirkulärer Import sonst unvermeidbar),
mit Re-Export für Rückwärtskompatibilität. VC-Auflösung ebenfalls dedupliziert (`c2_step_core`
duplizierte vorher `_resolve_vc`'s Branching inline).

**Sorgfältig geprüfter Fallstrick:** `action_logprobs_raw` hat für C1 aktuell eine flache Liste
(K-Sensitivity-Sweep/Sidecar-Pipeline erwarten das so), für C2 eine Liste von Listen (eine pro
Sample). Die geteilte Engine liefert immer die C2-Form; `c1_step_core`s Wrapper packt sie wieder
aus (`lp_saved[0]`), um den externen Vertrag exakt zu erhalten — keine stille Formatänderung an
einem bereits validierten Gate-C-Mechanismus.

**Verifiziert:** Volle Testsuite 361 passed (360 bestehend + 1 neuer C1-Regressionstest, der
beweist: unclosed `<think>` → leere Aktion, kein TLE, statt des literalen `'<think>'`-Strings).
Neuer Guard-Test stellt sicher, dass der Reasoning-Instruktionstext nur einmal im Quellbaum
definiert ist (Pendant zum bereits bestehenden Test für `_SINGLE_LINE_OUTPUT_INSTRUCTION`).
Lokaler Mock-Smoke über `scripts/gate_f_c1c2_quality_probe.py` bestätigt: C1 gibt jetzt korrekt
`""` zurück statt `'<think>'`, wenn Reasoning nicht schließt.

**Noch offen:** Realer Verifikationslauf auf dem Pod (Mock-Backend kann den `</think>`-Fall nicht
sauber simulieren, schließt seine Fake-Antworten nie). Gate-F-C1/C2-QC-Punkt bleibt bis dahin
offen — der Fix ist verifiziert, aber noch nicht am echten Modell bestätigt.

**Testsuite:** 361 passed.

## 2026-07-20 — C1/C2-Qualitätskontroll-Probe gelaufen (real, parallel): ToH/C1 sauber, ein echter C1-Parsing-Fund

**Zweck:** Gate-F-HART-Punkt "C1/C2-Qualitätskontrolle" — 5 Episoden/Zelle, real, gegen die
eingefrorenen Manifeste, via der parallelisierten `scripts/gate_f_c1c2_quality_probe.py`
(N=16 gleichzeitig, echter `EpisodeScheduler`/`run_phase1_job`-Pfad). Lief in 3523s (~59 min)
komplett durch: 20/20 Episoden completed, 0 failed.

**Ergebnis:**
- **ToH/C1: sauber, keine Befunde.**
- **ToH/C2 + TextWorld/C2:** je mehrere `thinking_unclosed`-Rejections nach 8192 Tokens — das ist
  der bereits vorhandene, korrekt funktionierende C2-Ablehnungsmechanismus (ein Kandidat, der sein
  `<think>` nicht rechtzeitig schließt, wird korrekt aus der Abstimmung ausgeschlossen statt
  fälschlich gezählt zu werden). Kein Bug, aber ein reales Signal, dass `cot_max_tokens=8192`
  gelegentlich zu knapp ist, besonders bei längeren, schwierigeren Situationen.
- **TextWorld/C1: 11 echte Befunde in 2 von 5 Episoden** (`qc_textworld_2_C1`, `qc_textworld_3_C1`).
  Das Modell verheddert sich in einer sich wiederholenden Denkschleife ("But the player has to
  proceed... This is a dead end..." endlos wiederholt, siehe Trace-Datei), schließt `</think>`
  nie, verbraucht das komplette 8192-Token-Budget. Anders als C2 hat C1s Parsing-Pfad **keine**
  `thinking_unclosed`-Ablehnung — er extrahiert stattdessen den literalen String `'<think>'` als
  "geparste Aktion", die die Umgebung dann als ungültigen Befehl ablehnt. Realer, aber begrenzter
  Fund: verschwendet einen Step pro Vorkommnis, keine systemische Korruption der Daten (die
  Umgebung lehnt den ungültigen Befehl einfach ab, der Schritt zählt als `illegal`). Noch nicht
  gefixt — bewusst für die nächste Session zurückgestellt (User ging schlafen, Fix an C1s
  Parsing-Pfad ist ein echter Code-Change, der Review braucht, kein Nacht-Alleingang).

**Voting-Korrektheit bei C2:** Kein einziger Fall, in dem der aufgezeichnete Sieger vom unabhängig
aus den rohen Stimmen neu berechneten Mehrheitsvotum abwich — das war der eigentliche Kernpunkt
der Qualitätskontrolle (funktioniert das Voting korrekt?) und ist damit sauber bestätigt.

**Daten:** 3.8 GB, 66 Dateien nach `data/results/gate_f_c1c2_quality_probe/` lokal gesynct
(gitignored, wie üblich).

**Offen:** C1-Parsing-Pfad um dieselbe `thinking_unclosed`-Behandlung wie C2 ergänzen (nächste
Session, mit Review). Gate-F-Checkbox bleibt bis dahin offen — der Fund ist real genug, um vor dem
finalen Abhaken behoben zu werden, auch wenn er die Voting-Korrektheit selbst nicht betrifft.

**Testsuite:** keine Quelländerung (nur Dokumentation).

## 2026-07-20 — Block-Aufteilung aus Gate F gestrichen (Nutzerentscheidung)

**Zweck:** Nach Vorlage der vier offenen Unterfragen (siehe Eintrag oben, "Vier Gate-Aufräumpunkte")
entschied der User, den WEICH-Punkt komplett zu streichen statt zu klären.

**Begründung (User, bestätigt):** Beide ursprünglichen Zwecke des Punkts — Resume-Grenzen und
Fehlerisolation — sind seit dem Gate-F-Resume-Fix (PR #23) bereits auf Episoden-Ebene gelöst
(atomares Schreiben, korrektes `--resume` unabhängig von Domain-Reihenfolge). Eine echte
Domain-Block-Trennung hätte einen neuen `--domains`-CLI-Flag gebraucht und das Risiko einer
N=32-Batch-Invarianz-Verletzung bei versehentlich parallelen Blöcken eingeführt — Mehraufwand ohne
erkennbaren Mehrwert gegenüber dem einfachen Beobachten/ggf. Killen+Resumen des laufenden
Single-Runs (bereits real getestet, `scripts/gate_f_resume_smoke.py`).

**Go/No-Go-Tabellenzeile Gate F aktualisiert:** Budget/Top-up jetzt korrekt als abgeschlossen
geführt (war noch als "offen" stehengeblieben), Block-Aufteilung als gestrichen vermerkt.

**Testsuite:** keine Quelländerung (nur Dokumentation).

## 2026-07-20 — Vier Gate-Aufräumpunkte: [VERIFY]-Zitate raus, Budget-Top-up erledigt, C1/C2-QC-Punkt neu, Block-Aufteilung geklärt

**[VERIFY]-Zitate:** Auf Nutzerwunsch komplett aus `blueprints/gate_p1_readiness.md` entfernt (Gate A,
war WEICH, blockierte ohnehin nichts). Die zugrundeliegende Zitat-Nacharbeit selbst (Scalena/EAGer,
Côté, Liu et al., Zhao-Erstautor) läuft weiter parallel zu Phase 1/2 im Thesis-Repo — nur nicht mehr
als Gate-Tracking-Punkt hier.

**Budget-Top-up:** Ursprüngliches "Offen: ~9 EUR Restguthaben"-Sub-Item war veraltet — User hat
zwischenzeitlich auf 50 EUR aufgeladen, aktuell ~36 EUR, und explizit erklärt: Zeit ist jetzt der
limitierende Faktor, nicht Geld. Als erledigt/akzeptiert markiert, zusammen mit der neuen
Token-basierten Budget-Zahl (~37–76.5h statt der alten ~48h, siehe Eintrag "Gate F: TextWorld-Cap
korrigiert..." oben).

**C1/C2-Qualitätskontrolle:** Neuer HART-Punkt unter Gate F. Begründung: Gate C-2 (Format-Compliance)
war ein kurzer Smoke, gelaufen vor der Gate-D-Kalibrierung UND vor dem `max_steps`-Fix — die jetzt
viel längeren, echten Episoden (TW Cap 45, ToH bis 45/Instanz) wurden nie mit echtem C1/C2-Reasoning
unter diesen Bedingungen geprüft. Bewusst mit der Run-Hygiene-Pod-Session gebündelt (User-Entscheidung),
kein separater Pod-Trip.

**Block-Aufteilung — Rechercheergebnis, offene Unterfragen:** Beim Nachschauen im Code zeigt sich, dass
"getrennte Blöcke" aktuell nicht direkt umsetzbar ist, ohne das vorher zu klären:

1. **Kein `--domains`-CLI-Flag.** `run_phase1.py`/`run_phase2.py` haben keinen Weg, eine einzelne
   Domäne gezielt zu laufen — nur `phase1.domains`/`phase2.domains` in der Config (Liste beider
   Domänen). Der `EpisodeScheduler` mischt beide Domänen in **einem** `ThreadPoolExecutor`-Lauf
   (`src/execution/worklist.py::build_phase1_worklist` baut eine gemeinsame Job-Liste, keine
   Domain-Reihenfolge-Garantie). Um echte, sequenzielle Blöcke zu bekommen, bräuchte es entweder
   einen neuen `--domains`-Flag oder manuelles Umschreiben der Config zwischen den beiden Läufen.
2. **Nebenläufigkeit zwischen Blöcken.** Falls zwei separate Prozesse (TextWorld-Block,
   ToH-Block) parallel liefen, würde die gemeinsame N=32-Batch-Invarianz-Grenze (C-1-Freeze)
   verletzt (bis zu 64 gleichzeitige Requests gegen denselben vLLM-Server). Blöcke müssen daher
   strikt sequenziell laufen, nie überlappend — sollte explizit festgehalten werden, ist aber kein
   Show-Stopper.
3. **Ist die Trennung noch nötig?** Der ursprüngliche Zweck ("Fehlerisolation") ist inzwischen
   teilweise schon durch den Gate-F-Resume-Fix gelöst (atomares Schreiben, korrektes Resume auf
   Episoden-Ebene — ein Crash in einer Domäne beschädigt die andere nicht). Was eine echte
   Domain-Trennung zusätzlich brächte: bewusst nach TextWorld pausieren und Kosten/Ergebnisse
   sichten, bevor die teurere ToH-Domäne (laut Budget-Neuschätzung der größere Kostentreiber)
   startet.
4. **Rerun-Kriterium unscharf.** `errors.jsonl`-Einträge (`append_episode_error()`,
   `src/execution/episode_runner.py`) enthalten nur einen freien Traceback-Text, kein
   strukturiertes "Infrastruktur vs. Content"-Feld. `classify_exclusion_reason()`
   (`src/utils/run_resilience.py`) kennt nur `env_assertion`/`label_error` (beides
   Content-/Logikfehler, keine Infrastruktur-Kategorie). §5.8 der Thesis-Prosa sagt nur
   "technisch fehlgeschlagene Episoden durch Infrastruktur werden dokumentiert und neu gelaufen"
   — ohne zu definieren, was als "Infrastruktur" zählt. Für die Praxis fehlt noch: eine klare
   Faustregel (z. B. Netzwerk-Timeout/Backend-Crash/OOM = Infrastruktur → Rerun; Modell antwortet
   nur falsch/parst nicht = valides Datum, kein Rerun-Grund).

**Nicht autonom entschieden** — dem User zur Klärung vorgelegt (siehe Chat), keine Code-Änderung in
diesem Eintrag.

**Testsuite:** keine Quelländerung (nur Dokumentation).

## 2026-07-20 — Gate E vollständig abgeschlossen: H3-Power-Simulation-WEICH-Punkt abgehakt

**Zweck:** User fragte nach einer Gesamtübersicht, welche Punkte in `blueprints/gate_p1_readiness.md`
jetzt abgehakt werden können. Beim Durchgehen: der WEICH-Punkt "H3-Power-Simulation" verlangt laut
eigenem Wortlaut "entweder durchführen oder die Limitation aktiv wählen und in §5.9 belassen" — beide
Pfade sind inzwischen genommen (Simulation durchgeführt und zweimal mit besserer Evidenz neu gelaufen;
§5.9-Absatz im Thesis-Repo ergänzt, Commit `77bfba9`, gepusht). Kein Nutzer-Entscheidungsrest mehr
offen, Checkbox abgehakt.

**Ergebnis:** Gate E steht damit auf "abgeschlossen" (beide HART-Punkte + der einzige WEICH-Punkt
erledigt). Restliche offene Punkte im Dokument bleiben unverändert offen — echte, nicht-technische
Gründe: Gate D (Freeze-Tag, wartet strukturell auf Gate F), Gate F (Run-Hygiene braucht Live-Pod-
Verifikation; Block-Aufteilung noch nicht umgesetzt; Budget-Top-up ist eine Finanzentscheidung des
Users), Gate A (offene [VERIFY]-Zitate, Literatur-Task im Thesis-Repo, blockiert nichts).

**Testsuite:** keine Quelländerung (nur Dokumentation).

## 2026-07-20 — Run-Hygiene: Modell-Speicherort korrigiert, Preflight-Script gebaut

**Zweck:** User korrigierte den gerade reparierten Run-Hygiene-Punkt: das Modell liegt **nie** auf
dem persistenten Network Volume, immer auf der ephemeren Container-Disk. Der wiederhergestellte
Volltext hatte fälschlich "Modell vorab auf dem Network Volume" übernommen — Verifikation gegen
`docs/runpod.md` und `scripts/pod_runtime_env.sh` bestätigt: `HF_HOME=/root/.cache/huggingface`
(ephemer), bewusst nicht `/workspace` (RunPods Template-Default zeigt dort fälschlich hin, wird von
`pod_runtime_env.sh` korrigiert). Network Volume soll klein bleiben (nur Code + Results,
"10 GB is enough"), Re-Download des Modells nach Container-Neustart ist einkalkuliert.

**Weiterer Fund währenddessen:** `RESULTS_DIR`-Env-Var wird von keinem Script tatsächlich gelesen —
reine Dokumentationskonvention. Der eigentliche Persistenz-Mechanismus ist, dass `--checkpoint-dir`
relativ zum Repo-Root defaultet, und das Repo selbst unter `/workspace` liegt. Checkliste entsprechend
korrigiert, damit sie den echten Mechanismus beschreibt statt der Env-Var.

**Preflight-Script gebaut:** `scripts/gate_f_run_hygiene_preflight.py` — fünf Checks ohne GPU/
Inferenz, lokal in Sekunden lauffähig:
1. `HF_HOME` nicht unter `/workspace` + Modell+Revision tatsächlich im lokalen Cache vorhanden.
2. Langfuse: entweder explizit aus, oder Credentials + SDK vorhanden.
3. Repo-Checkout selbst liegt unter `/workspace`.
4. History-Guard: keine Truncation-Parameter aktiv (spiegelt `src/utils/history_guard.py`'s eigene
   Laufzeitprüfung, schlägt hier aber vorab fehl statt mitten im Batch).
5. Sidecar-Mode `action_window`, Full-Instanzen kollidieren nicht mit Holdout, `execution.
   max_concurrent_episodes=32` (C-1-Freeze).

**Lokal getestet** (Mac, kein Pod): History-Guard und Sidecar/Concurrency-Checks laufen sauber grün
(bestätigt, dass die aktuelle `experiment_core.yaml` an diesen Punkten schon korrekt ist); die drei
pod-spezifischen Checks (HF_HOME, Langfuse, `/workspace`) schlagen erwartungsgemäß fehl, da lokal
kein Pod-Environment vorliegt — Logikpfade einzeln mit simuliertem `HF_HOME` verifiziert (echter
Cache-Treffer und `/workspace`-Fehlschlag beide korrekt erkannt).

**Testsuite:** 358 passed (kein Produktionscode geändert, nur ein neues Dev-Script + Doku).

## 2026-07-20 — Run-Hygiene-Checklisten-Punkt: Dokubug seit 2026-07-14 gefixt

**Zweck:** User fragte nach dem nächsten Schritt für die Gate-F-Run-Hygiene-Checkliste; beim
Nachlesen fiel auf, dass der HART-Punkt selbst kaputt war.

**Bug:** Commit `5be09f7` (2026-07-14, "feat(logging): add action-window logprob sidecar modes")
sollte nur die veraltete Sidecar-Aussage ("Sidecars aus für volle Phase 1/2") durch die neue
Drei-Stufen-Policy ersetzen, hat dabei aber versehentlich den kompletten übrigen Satz (Network
Volume, Langfuse-Keys, `RESULTS_DIR`, History-Guard, Backend-Einstellungen) durch zwei
Ellipsen-Platzhalter ("…") ersetzt statt sie stehen zu lassen. Seit sechs Tagen unbemerkt, weil der
Punkt ohnehin unangehakt und nie einzeln gelesen wurde. Keine Rogue-Agent-Aktion — echter
Editier-Fehler in einem regulären Feature-Commit, per `git log -p` gefunden.

**Fix:** Ursprünglichen Volltext (aus dem Vorgänger-Commit `22fd887`) mit der neuen
Sidecar-Policy-Formulierung zusammengeführt: Network Volume, Langfuse/Tracing-Dokumentation,
`RESULTS_DIR`, History-Guard, `logprob_sidecar_mode: action_window` +
`logprob_sidecar_full_instances`, batch-invariantes Backend, Evidenz `run_metadata` des ersten
Phase-1-Blocks.

**Inhaltlich unverändert:** Dieser HART-Punkt lässt sich nicht vorab lokal smoken wie
Resume-Korrektheit — die Evidenz ist per Definition das `run_metadata.json` des tatsächlichen ersten
Phase-1-Blocks auf dem Pod. Empfehlung: vor dem vollen 48h-Lauf einen kurzen Pod-Preflight (Modell
geladen, Secrets/Tracing-Entscheidung, `RESULTS_DIR`, Sidecar-Config, N=32-Batch-Invarianz) als
Mini-Smoke fahren statt die Prüfung erst live im echten ersten Block zu machen.

**Testsuite:** keine Quelländerung (nur Dokumentation).

## 2026-07-20 — ToH-Seite der H3-Power-Simulation mit echten Freeze-Korridor-Längen neu gelaufen

**Zweck:** `docs/gate_e_h3_power_simulation.md` (Abschnitt 3) flaggte explizit, dass die ToH-Seite
der Simulation die rohe, nicht-schwierigkeitskalibrierte Gate-C-Pilotlängen-Verteilung (Ø≈18 Steps,
viele nahe altem 20-Step-Cap) als Bootstrap-Pool nutzte, mit dem Vorbehalt: "echte ToH-Episoden
könnten am Ende kürzer ausfallen" — die ToH-Zahlen seien daher "ein optimistischer Kontextwert,
nicht die belastbarste Aussage". Nach dem ToH-Freeze (2026-07-19: 4 Disks, C1-Referenz,
`random_scramble`) lag der reale Vergleichswert vor; neu gelaufen, um den Vorbehalt entweder zu
bestätigen oder zu widerlegen statt ihn stehen zu lassen.

**Änderung:** `scripts/h3_power_simulation.py` — neue Funktion `load_toh_frozen_corridor_lengths()`
liest echte Episodenlängen aus `data/results/gate_d_calibration/toh_corridor_scramble_n30/`
(derselbe n=30-Lauf, der schon die Korridor-Entscheidung selbst trug), filtert auf C1 + 4 Disks
(Disk-Zahl steht nicht als eigenes Feld im Episode-JSON, sondern wird per Regex aus dem
Judge-Prompt-Text "holds all N disks" extrahiert — verifiziert gegen die dokumentierte
Erfolgsrate: 6/15 = 40,0 %, exakt der in der Freeze-Entscheidung berichtete Wert, also korrekt
gefiltert). `build_design_cells()` nutzt diesen Pool jetzt für ToH statt der alten Pilotlängen, per
neuem optionalem `--toh-length-run-dir`/`--toh-length-num-disks`-CLI-Flag (Default: altes Verhalten,
Rückwärtskompatibilität zu v1/v2 erhalten). TextWorld-Seite unverändert.

**Ergebnis** (`data/results/gate_e_h3_power/h3_power_simulation_v3.json`, n=15 echte Längen, Ø 28,4
Steps, Range 3–45 — **länger**, nicht kürzer, als die alte Pilot-Annahme):

| | ToH/TLE Power bei β_int=−0,15 (α=.05) | (α=.025) | 80-%-Schwelle \|β_int\| (α=.05) |
|---|---:|---:|---:|
| Alt (Pilot-Längen, Ø≈18) | 51,0 % | 41,8 % | 0,210 |
| Neu (Freeze-Korridor, Ø 28,4) | **71,8 %** | **60,8 %** | **0,175** |

**Befund:** Der Vorbehalt hat sich als falsch herum erwiesen — die echten, kalibrierten 4-Disk-C1-
Episoden sind im Schnitt länger als die alte Pilot-Verteilung (mehr Fehlversuche/Umwege vor Erfolg
oder Abbruch am Cap 45, nicht nur die reine Optimalzug-Länge von 15), was mehr Positionsauflösung
und damit mehr Power liefert, nicht weniger. Die ToH-Zahlen sind damit keine Überschätzung mehr,
sondern eine reale, korridor-treue Schätzung. TextWorld bleibt weiterhin der primäre,
konfirmatorische Befund; ToH bleibt exploratorisch (§5.8/§5.9 der Thesis-Prosa), dieser Rerun ändert
daran nichts, verbessert nur die Kontext-Zahlengrundlage.

**Bekannte Einschränkung, unverändert:** n=15 ist ein kleiner Resampling-Pool (wenige diskrete
Werte) — kleiner als der ursprüngliche gepoolte Pilot-Pool, aber zielgenau auf die tatsächlich
eingefrorene Konfiguration statt großzügig auf eine veraltete.

**Thesis-Prosa (`../metacog-thesis`):** Der neue §5.9-Absatz (siehe unten, Eintrag "H3-Power-
Simulation..." — separat, im Thesis-Repo committed) erwähnt ToH nur qualitativ ("keine vergleichbare
Power-Garantie"), keine ToH-Zahlen zitiert — bleibt nach diesem Rerun unverändert korrekt, keine
Nachbesserung nötig.

**Testsuite:** keine Produktionscode-Berührung (reines Analyse-Skript); `python -m pytest -q`
unverändert 358 passed.

## 2026-07-20 — Gate F: Resume-Korrektheit unter Nebenläufigkeit getestet, echter Bug gefunden und gefixt (PR #23)

**Zweck:** Gate-F-HART-Punkt "Resume-Korrektheit unter Nebenläufigkeit" — laufenden gebatchten Smoke
hart abbrechen, mit `--resume` fortsetzen, prüfen: kein halb geschriebenes Episode-JSON, kein
Doppel-Eintrag, kein übersprungenes Work-Item über `list_completed_episodes`.

**Pod nötig?** Nein — geprüft und lokal mit Mock-Backend durchgeführt. Die Korrektheitseigenschaft
hängt ausschließlich an der Bookkeeping-/Threading-Logik des Harness (`ThreadPoolExecutor` in
`src/execution/scheduler.py`, Dateisystem-Glob-basiertes Resume in `list_completed_episodes()`),
nicht am Backend, das die Episoden-Inhalte liefert. Der Mock-Backend macht das Race sogar leichter
reproduzierbar (mehr Episoden pro Sekunde als reales vLLM).

**Bug gefunden:** `log_episode()` (`src/utils/logging_utils.py`) schrieb `json.dump()` direkt auf
den finalen `ep_*.json`-Pfad, ohne Temp-Datei + Rename. Ein `SIGKILL` mitten im Schreiben hinterließ
eine abgeschnittene Datei; `list_completed_episodes()` prüft nur Dateiexistenz (Glob), nicht
Inhaltsvalidität — die korrupte Episode wäre bei `--resume` für immer als "fertig" gezählt und nie
neu gelaufen. In echtem Phase-1/2-Betrieb hätte das Episoden stillschweigend aus dem Datensatz
verschwinden lassen, ohne Fehlermeldung.

**Reproduktion (nicht nur Herleitung):** Echtes `SIGKILL` gegen einen laufenden
`scripts/run_phase1.py`-Batch (`scripts/gate_f_resume_smoke.py`, neues Script, wiederverwendbar) —
2 unabhängige Batch-Zyklen produzierten je eine real korrupte, dauerhaft "stuck" Episodendatei.
Zusätzlicher gezielter Write-Race-Probe (großes synthetisches Payload weitet das Schreibfenster):
7/10 Treffer.

**Fix:** Schreiben auf eine Sibling-Temp-Datei, dann `os.replace()` auf den finalen Pfad (atomar
unter POSIX/Windows), mit Cleanup bei nicht-fatalen Schreibfehlern. Nach Fix: 0/8 reale
Hard-Kill-Zyklen und 0/10 Probe-Trials zeigen die Race noch. Regressionstest ergänzt
(`tests/test_checkpointing.py`), erzwingt einen Schreibfehler und prüft, dass weder der finale Pfad
noch `list_completed_episodes()` davon betroffen sind.

**Testsuite:** 358 passed (357 + neuer Regressionstest). PR #23, gemerged `dd38e44`.

**Gate-F-HART-Punkt "Resume-Korrektheit" abgehakt.** Noch offen: Run-Hygiene-Checkliste,
RunPod-Budget-Top-up-Entscheidung.

## 2026-07-20 — TextWorld-Difficulty-Sweep-HART-Punkt abgehakt; Freeze-Tag-Timing geklärt

**Zweck:** User fragte, warum die TextWorld-Sweep-Box in `blueprints/gate_p1_readiness.md` trotz
final eingefrorener Manifeste noch offen war, und bat, den Freeze-Tag zu setzen und alle Gate-D-
Punkte abzuhaken.

**TextWorld-Sweep-Box:** Reine Inkonsistenz, kein Sachgrund — die Korridor-Bestätigung existiert
bereits seit 2026-07-18 (n=16, Seed 9001, identische Generierungsparameter wie das finale 50er-
Manifest Seed 20260718): `r5_i1_take+cook` bei 43.75 % Erfolg, im 30–50-%-Korridor. Exakt dieselbe
Evidenzqualität (Validierungssample statt Lauf auf den finalen Instanzen selbst), mit der die ToH-
Zeile am 2026-07-19 bereits abgehakt wurde. Jetzt nachgezogen, abgehakt.

**Freeze-Tag:** Kein Nachziehen, sondern eine echte Timing-Frage — dem User zur Bestätigung
vorgelegt (`AskUserQuestion`), nicht autonom entschieden. Laut Dokument selbst (`gate_p1_readiness.md`,
Zeilen 19 + 115) gibt es einen einzigen projektweiten Freeze-Tag, gesetzt nach Gate F, nicht pro
Gate — Gate F hat mit dem Resume-Correctness-under-Concurrency-Test noch einen offenen HART-Punkt.
User bestätigte: Tag bleibt bis nach Gate F offen, kein Vorziehen. Entsprechend bleibt die letzte
Gate-D-Zeile ("Beide Manifeste final und im Freeze-Tag") formal offen, mit korrigierter Begründung
(Tag-Timing statt "bewusst noch nicht gesetzt" ohne Kontext). Go/No-Go-Tabellenzeile für Gate D auf
"teilweise" aktualisiert.

**Testsuite:** keine Quelländerung (nur Dokumentation).

## 2026-07-20 — Gate E gegen aktuellen Code re-verifiziert, beide HART-Punkte abgehakt

**Zweck:** Vor dem formalen Abhaken der Gate-E-Checkboxen geprüft, ob die am 2026-07-17
dokumentierte Rehearsal-Evidenz nach den seitdem gelandeten Fixes noch stimmt — nicht einfach
angenommen. Der "Rückwirkungs-Check" vom 2026-07-17 (`docs/consistency_log.md`, Eintrag
"Rückwirkungs-Check: Gate-E-Bugfixes betreffen keine bereits berichteten Gate-D/C-Ergebnisse") deckte
nur die drei **während** des Rehearsals gefundenen Bugs ab (extends-Overlay, YAML-`off`,
Phase-2-Episoden-Drop) — nicht den `cluster_bootstrap`-NaN-Sortierbug, der **später am selben Tag**
im Housekeeping-Pass gefunden wurde und laut dessen eigenem Log-Eintrag "rückwirkend die im selben
Tag im Gate-E-Rehearsal als 'unklares Kleine-Cluster-Phänomen' notierte Anomalie erklärt" — dieser
Rückschluss wurde nie durch einen tatsächlichen Neu-Lauf verifiziert.

**Nachgeholt:** `scripts/gate_e_rehearsal.py` mit identischen Eingabedaten
(`data/results/instrument_validation/phase1_20260714_105004/`, `--holdout-instances 3`) gegen den
aktuellen Code neu laufen lassen. Ergebnis: Die TextWorld-Anomalie ist weg — CI weitet sich korrekt
auf `[-0.0195, 0.1688]` (535/5000 nicht-endliche Replikate jetzt korrekt gefiltert statt die
Perzentile zu verfälschen), Punktschätzer (0,0796) liegt jetzt innerhalb des CI, `skewness=-0,402`
statt `NaN`. Pooled/ToH-Zeilen unverändert (0 nicht-endliche Replikate, nie betroffen). Volles
Detail: `docs/gate_e_rehearsal.md`, Update-Hinweis oben im Dokument.

**Pre-Analysis-Screen-HART-Punkt:** keine der seit 2026-07-17 gelandeten Fixes berührt
`preanalysis_screen.py` — unverändert gültig, keine Neu-Verifikation nötig.

**Ergebnis:** Beide Gate-E-HART-Punkte in `blueprints/gate_p1_readiness.md` abgehakt. Der
WEICH-Punkt (H3-Power-Simulation) bleibt bewusst unangehakt — nicht weil Arbeit fehlt (durchgeführt
und am 2026-07-19 mit der korrigierten Längen-Annahme aktualisiert), sondern weil er laut eigener
Formulierung eine Nutzer-Entscheidung verlangt (wie mit dem Power-Befund im Kapitel umgegangen wird),
keine technische Restarbeit.

## 2026-07-19 — ToH-Manifest final eingefroren: 4 Disks, C1-Referenz, `random_scramble`

**Zweck:** Abschluss des Gate-D-ToH-Prozesses (siehe die zahlreichen Einträge der letzten Tage:
Peg-C-Vermeidungs-Bias, Prompt-Fixes, State-Diversitäts-Proben, Korridor-Läufe). Dieser Eintrag
dokumentiert die finale Entscheidung und den Freeze.

**Vorgeschichte in Kürze:** C0 zeigte über drei unabhängige, echte Sweeps (n=10 gemischt 3+4 Disks,
n=20 gemischt zufällig, n=30 gemischt `random_scramble`) konsistent ~0–17 % Erfolg — kein
Schwierigkeitsproblem, sondern ein reproduzierbarer, per Trace-Analyse belegter Bias (Modell wählt
den als Ziel benannten Peg nie als Zugquelle). Drei Prompt-Klarstellungen (Zielzustand statt
Bewegungsanweisung, Disk-Größen-Erklärung, neutrales Format-Beispiel, plus ein Bugfix für ein
domänenfremdes "go north"-Beispiel im geteilten Prompt-Template) haben den Bias abgeschwächt, aber
nicht beseitigt — C0 bleibt kein Referenzkandidat. Korridor daher wie schon am 2026-07-17 vermutet
an **C1** kalibriert.

**Zwei strukturelle Bugs unterwegs gefunden und gefixt** (beide vor dem Freeze kritisch):
1. Die ursprüngliche Instanzgenerierung (`partial_start_mode="optimal_prefix"`, deterministischer
   Einzelpfad) besuchte nur `num_disks+1` Zustände — unter der Produktionsverteilung (Disks {3,4},
   Partial-Start [0,3]) exakt **8 unterscheidbare Puzzles**, unabhängig davon, wie viele "Instanzen"
   generiert wurden. Neuer Modus `partial_start_mode="random_scramble"` zieht ohne Zurücklegen aus
   dem vollen, per BFS verifizierten 3ⁿ-Zustandsraum (27/81/243 für 3/4/5 Disks).
2. `make_experiment_env()` rekonstruierte ToH-Instanzen zur Laufzeit immer mit dem alten
   `optimal_prefix`-Standard, unabhängig vom Manifest — ein mit `random_scramble` gebautes Manifest
   wäre zur Laufzeit stillschweigend auf den alten 8-Puzzle-Zustandsraum zurückgefallen. Gefixt:
   Manifest speichert `num_disks_range`/`partial_start_range`/`partial_start_mode` jetzt pro Eintrag
   (analog zum bestehenden `task_generation_seed`-Muster), `make_experiment_env()` liest diese
   bevorzugt aus dem Manifest. End-to-End verifiziert (identischer Zustand bei Rekonstruktion).

**Korridor-Entscheidung — bewusst nur an Erfolgsrate, nicht an AUROC.** Wichtiger methodischer
Punkt, der während der Session explizit diskutiert wurde: Die Konfigurationswahl (welche Disk-Zahl
einzufrieren ist) darf sich **nicht** an der späteren AUROC/Signalqualität orientieren — das wäre
eine Form von Selektion auf die abhängige Variable ("Garden of Forking Paths"), selbst wenn die
finalen 50 Instanzen aus einer frischen, unabhängigen Ziehung stammen. Ein zwischenzeitlich erwogener
Filter ("3-Disk-Instanzen mit hoher `optimal_steps`, um gleichzeitig Korridor und gute AUROC zu
treffen") wurde deshalb **verworfen** — unabhängig davon, dass er sich auch empirisch als Sackgasse
erwies (der 3-Disk-Zustandsraum deckelt bei 7 Optimalzügen; C1 löst selbst die am weitesten
entfernten 3-Disk-Zustände zu 75 %, kein nutzbares Schwierigkeitsgefälle).

**Finale Konfiguration:** 4 Disks (fix, nicht {3,4} gemischt — 3 Disks separat getestet: 73 % Erfolg,
deutlich zu leicht, siehe oben), `partial_start_mode=random_scramble`. Isolierte 4-Disk-Messung
(n=15, Teilmenge eines n=30-Laufs, Seed 3141): **C1-Erfolg 40 %** — sauber im 30–50-%-Korridor.
C0 zum Vergleich: 13–20 % je nach Schnitt, weiterhin kein Kandidat.

**Freeze:** `scripts/build_toh_manifest.py --num-instances 50 --seed 271828 --holdout-count 5
--holdout-policy mod-10 --num-disks-range 4 4 --partial-start-mode random_scramble`. Neuer Seed,
disjunkt von allen Kalibrierungs-Seeds (42 alter Sweep, 2026 n=20-Lauf, 3141 n=30-Lauf). Verifiziert:
**50 von 50 Instanzen sind echt unterschiedliche Zustände** (vorher wären es maximal 8 gewesen).
`optimal_steps`-Spanne 3–15 (Ø 10,2) — reale Schwierigkeitsstreuung. Holdout mod-10 (Instanzen
0/10/20/30/40), analog zu TextWorld.

**Smoke:** `scripts/gate_d_manifest_smoke.py` war hart auf C0 verdrahtet (hätte für ToH mit
C1-Referenz immer `Hard-GO: False` geliefert) — gefixt via neue `REFERENCE_STAGE_BY_DOMAIN`-Map
(`textworld: C0, tower_of_hanoi: C1`). Mock-Lauf über alle 50 Instanzen: `holdout`/`difficulty_tier`
kommen korrekt an, Referenzstufe korrekt C1. Kein echter Realmodell-Lauf über die vollen 50 gemacht
(bewusst — die Erfolgsraten-Frage ist über die vier vorherigen echten Sweeps bereits hinreichend
beantwortet; ein weiterer Lauf hätte hier nur AUROC-Neugier bedient, die laut obigem Prinzip nicht
die Konfigurationswahl treiben soll).

**Testsuite:** 357 passed (356 + neuer Test für `REFERENCE_STAGE_BY_DOMAIN`).

**Offen:** Freeze-Tag (git tag) selbst noch nicht gesetzt — bewusst Usersache, keine technische
Restarbeit. Blueprint-Checkboxen entsprechend aktualisiert, aber "Beide Manifeste final" bleibt
unangehakt, bis der Tag gesetzt ist.

## 2026-07-19 — H3-Power-Simulation mit korrigierter Episodenlängen-Annahme neu gelaufen

**Zweck:** Die H3-Power-Simulation vom 2026-07-17 (`docs/gate_e_h3_power_simulation.md`) hatte für
TextWorld eine uniforme Episodenlänge **8–15 Steps** angenommen — genau die A-priori-Vorgabe, die
am 2026-07-18 durch die realen Post-Fix-Sweeps auf **15–40 Steps** revidiert wurde (siehe Eintrag
"TextWorld-Episodenlängen-Vorgabe korrigiert" weiter unten). Die Power-Simulation nutzte damit eine
seit gestern bekannt überholte Annahme. Neu gelaufen mit der aktuellen Vorgabe.

**Änderung:** `scripts/h3_power_simulation.py` — `length_mode` für TextWorld von `"uniform_8_15"`
(`rng.integers(8, 16)`) auf `"uniform_15_40"` (`rng.integers(15, 41)`) umgestellt, Rationale-Text
in `build_design_cells()` entsprechend aktualisiert. ToH-Seite (Bootstrap aus Pilot-Längen)
unverändert. Sonst identisches Simulationsdesign (400 Replikate/Zelle primär, 250 sekundär, echte
`fit_h3_model`-GEE-Fits, Seed `20260717`).

**Ergebnis** (`data/results/gate_e_h3_power/h3_power_simulation_v2.json`), TextWorld/TLE (der
zentrale konfirmatorische Fall):

| $\beta_{int}$ | Power alt (8–15, α=.05) | Power neu (15–40, α=.05) |
|---:|---:|---:|
| −0,10 | 21,5 % | **33,3 %** |
| −0,15 (moderate Degradation, ~50 % Abschwächung) | 34,8 % | **51,3 %** |
| −0,20 | 52,3 % | **73,3 %** |
| −0,30 | 79,0 % | **97,8 %** |

**80-%-Power-Schwelle:** vorher $|\beta_{int}|\approx0{,}307$ (≈102–120 % von $\beta_z=0{,}30$,
praktisch vollständiger Signalverlust bis Episodenende nötig) → jetzt
**$|\beta_{int}|\approx0{,}228$ (≈76 % von $\beta_z$)**. Bei moderater Degradation steigt die Power
von ~35 % auf ~51 % (α=.05) bzw. ~23 % auf ~42 % (α=.025, Holm-konservativ) — ein substanzieller,
kostenloser Gewinn allein aus der bereits beschlossenen Längen-Korrektur, kein neuer Trade-off.

**Einordnung:** Ändert die grundsätzliche Aussage aus dem 07-17-Report nicht (Interaktionstests
bleiben strukturell schwerer zu entdecken als Haupteffekte, moderate Degradation bleibt
unterpowert), verbessert aber die konkrete Zahlengrundlage substanziell. Der 07-17-Report
(`docs/gate_e_h3_power_simulation.md`) bleibt als historisches Dokument stehen, wird aber um einen
Verweis auf diesen aktualisierten Lauf ergänzt, damit niemand versehentlich die überholten
8–15-Zahlen zitiert.

**Testsuite:** keine Quelländerung außerhalb von `scripts/h3_power_simulation.py` (reines
Analyse-Skript, keine Produktionscode-Berührung); `python -m pytest -q` unverändert 356 passed.

## 2026-07-18 — TextWorld-Instanzen final generiert und Manifest gebaut (Gate D, TextWorld-Teil)

**Zweck:** Nach der n=16-Bestätigung (Eintrag unten) hat der User `r5_i1_take+cook` als finale Zelle
gewählt (Cooking-Variante näher am "normalen" Spieldesign, bei ohnehin gleichwertigen Zahlen).
Dieser Eintrag dokumentiert die Umsetzung: die tatsächlichen 50 produktiven Instanzen + Manifest.

**Generierung:** `scripts/generate_textworld_games.py --num-rooms 5 --num-ingredients 1 --cook
--seed 20260718 --num-instances 50 --output-dir data/tasks/textworld`. Seed bewusst neu gewählt
(weder Sweep-Seed 42 noch Bestätigungs-Seed 9001), damit die finalen Instanzen unabhängig von der
Kalibrierung sind. **Überschreibt 5 alte Dev-Fixture-Instanzen** (April, `num_ingredients=2`, auf
die mehrere generische Dev/Smoke-Configs zeigen — nicht git-getrackt, `data/tasks/` ist komplett
gitignored, jederzeit neu generierbar; User hat das Überschreiben explizit bestätigt).

**Manifest:** `scripts/build_textworld_manifest.py --holdout-count 5 --holdout-policy mod-10`.
`mod-10` (Spread-Holdout: Instanzen 0/10/20/30/40) statt `first-n` gewählt — die Thesis-Prosa legt
die Auswahlpolitik nicht fest, Spread vermeidet den Anschein eines willkürlichen zusammenhängenden
Blocks. `data/tasks/textworld/difficulty_manifest.json`: 50 Instanzen, `holdout_count=5,
non_holdout_count=45`, `difficulty_tier`-Verteilung 30× easy / 20× medium (0× hard) — **das ist die
Solver-Walkthrough-Länge** (Ø 7.98, Max 10 Steps optimal), nicht das empirische Modellverhalten; die
beiden Zahlen sind unabhängig (siehe Rückfrage des Users dazu) — der Median-Sieg-Step aus der
n=16-Bestätigung (15) bleibt die relevante Zahl für die H3-Längen-Diskussion, nicht `difficulty_tier`.

**Smoke (`scripts/gate_d_manifest_smoke.py`, Blueprint-Punkt "holdout/difficulty_tier im
Episode-JSON"):** Erster Lauf schlug fehl — `create_execution_backend(config, use_real_model)` rief
die Factory positional statt mit dem laut Signatur (`*, use_real: bool`) erzwungenen Keyword auf;
jeder andere Aufrufer im Repo nutzt bereits `use_real=...`. Nie zuvor ausgeführtes Script, daher nie
aufgefallen. **Gefixt** (`create_execution_backend(config, use_real=use_real_model)`), Regressionstest
`tests/test_gate_d_manifest_smoke.py` ergänzt (patcht die Factory und prüft den Keyword-Aufruf sowie
dass `holdout`/`difficulty_tier` aus einem synthetischen Manifest-Eintrag im Report ankommen — ohne
echte Spiel-/Modell-Abhängigkeit). Danach mit Mock-Modell über `--domains textworld` (ToH-Manifest
existiert noch nicht, wäre sonst mit `FileNotFoundError` abgebrochen) auf allen 50 Instanzen
gelaufen: `holdout`-Flags (0/10/20/30/40) und `difficulty_tier`-Werte kommen korrekt im
Episode-Report an. **Bewusst kein echter Realmodell-Lauf über alle 50** (das wäre eine zusätzliche
volle Korridor-Rekonfirmation auf derselben Parameterkombination, die die n=16-Bestätigung bereits
unabhängig erbracht hat — separate Kosten-/Zeitentscheidung, nicht Teil dieser Plumbing-Prüfung).

**Testsuite:** 353 passed (352 + neuer Regressionstest).

**Offen:** ToH-Konfiguration/Manifest weiterhin nicht final — der Gate-D-HART-Punkt "Beide Manifeste
final" bleibt bis dahin offen; nur der TextWorld-Teil ist jetzt fertig. Blueprint-Checkbox bewusst
nicht angehakt (Gate-Status-Entscheidungen sind Usersache, siehe Konvention in diesem Log).

## 2026-07-18 — TextWorld-Korridor-Kandidaten bei n=16 bestätigt: 3 von 4 halten, einer fällt raus

**Zweck:** Die 4 Kandidatenzellen aus dem n=4-Sweep (2026-07-16, Cap 45: `r5_i1_take-only` als
`best_candidate`, plus die 3 `corridor_candidates` `r5_i1_take+cook`, `r3_i2_take-only`,
`r5_i3_take-only`) waren zwischen Cap-45- und Cap-70-Lauf nicht stabil (n=4/Zelle zu klein/verrauscht,
siehe Eintrag "2026-07-16 — Post-Fix-Sweeps"). Bestätigung mit unabhängiger Neu-Stichprobe (n=16,
eigener Seed statt Wiederverwendung der ursprünglichen 4 Instanzen — echte Replikation, kein
Extend) via `scripts/validate_textworld_candidate.py` (existierte bereits seit `db5b482`, aber nie
zuvor ausgeführt), auf dem 5090-Pod, `--real`, Cap 45.

**Ergebnis** (`data/results/gate_d_calibration/textworld_candidate_confirmation/validation_results.json`):

| Zelle | success@Cap (n=16) | Korridor (30–50%) | median_win_step | mean_len_success | Trunkierung |
|---|---|---|---|---|---|
| `r5_i1_take-only` | 0.375 | ✅ | 15.0 | 16.0 | 43.8% |
| `r5_i1_take+cook` | 0.4375 | ✅ | 15.0 | 16.4 | 25.0% |
| `r3_i2_take-only` | 0.4375 | ✅ | 14.0 | 16.9 | 37.5% |
| `r5_i3_take-only` | **0.25** | ❌ | 13.5 | 16.5 | 56.3% |

**Befund:** 3 von 4 Zellen bestätigen sich robust im Korridor (37.5–43.75%, sauber um die Mitte
zentriert). `r5_i3_take-only` — bei n=4 noch bei 50% und als einzige Zelle "inside_length_guidance"
unter der alten 8–15-Vorgabe — fällt bei n=16 auf 25% und damit unter den Korridor. Genau die
Instabilität, die der n=4-Vergleich befürchten ließ, jetzt mit echten Daten aufgelöst statt vermutet.

**Bezug zur Längen-Revision (siehe Eintrag "TextWorld-Episodenlängen-Vorgabe korrigiert" unten):**
Alle 3 überlebenden Zellen liegen mit `median_win_step` 14–15 knapp **an**, nicht komfortabel über
der neuen 15-Step-Untergrenze — ein Punkt, den man bei der finalen Zellwahl im Blick behalten sollte
(knapper Puffer zur H3-load-bearing-Grenze), aber `mean_episode_length_success` (16.0–16.9) liegt
etwas darüber.

**Offen (User-Entscheidung, nicht autonom getroffen):** finale Wahl unter den 3 verbleibenden Zellen
für den Gate-D-Freeze (50-Instanzen-Generierung). Reiner Zahlenvergleich: `r5_i1_take+cook` hat die
niedrigste Trunkierungsrate (25%) bei vergleichbarem Erfolg; die drei sind bei n=16 im Rahmen des
erwartbaren Rauschens praktisch gleichwertig, kein Kriterium sticht klar heraus.

**Testsuite:** keine Quelländerung (nur Datenlauf über bestehendes, ungetestetes-aber-vorhandenes
Script); kein neuer pytest-Lauf nötig.

## 2026-07-18 — TextWorld-Episodenlängen-Vorgabe korrigiert: 8–15 → 15–40 Steps (A-priori-Zahl durch Post-Fix-Sweep-Daten widerlegt)

**Korrektur zu:** Der in `blueprints/gate_p1_readiness.md` (Gate D, HART-Punkt "TextWorld-Difficulty-Sweep") und `docs/textworld.md` gesetzten Zielvorgabe "8–15 Steps mittlere Episodenlänge", wortgleich zitiert in der Thesis-Prosa (`../metacog-thesis/chapters/05_methodology.md` §5.5.1, dort bislang mit `[PENDING: difficulty calibration]` markiert) und `chapters/outline.md` §5.5.

**Ursprung der alten Zahl:** reine A-priori-Setzung ohne empirische Grundlage (keine Sweep-Daten zum Zeitpunkt der Präregistrierung verfügbar) — vermutlich als "genug Positionsauflösung für H3, aber nicht zu lang" gedacht, nie gegen echte Modelldaten geprüft.

**Widerlegt durch:** die echten Post-Fix-Sweeps (siehe Eintrag "2026-07-16" weiter unten, Config-Wiring-Bug-Fix). `data/results/gate_d_calibration/textworld_sweep/sweep_results.json` (Cap 45): `global_p90_win_step_success = 41.8`; die Zellen, die den 30–50-%-Erfolgskorridor erfüllen (`corridor_candidates` + `best_candidate`), haben `metrics.mean_episode_length_success`-Werte von **11.5, 13.5, 23.0, 41.5** — nur 2 von 4 liegen überhaupt in [8,15], und `ranked_results` zeigt insgesamt, dass praktisch alle Zellen mit realistischem Erfolg (Erfolgsrate ≥ 25 %) bei Sieg-Längen zwischen 5 und 43 Steps liegen, mit einem klaren Schwerpunkt oberhalb von 15. Die Kontroll-Sweep bei Cap 70 (`textworld_sweep_cap70/sweep_results.json`, identische Instanzen/Seed) bestätigt sowohl den fehlenden Floor-Effekt (`global_p90_win_step_success = 40.6`, Trunkierungsrate ähnlich zu Cap 45) als auch dieselbe n=4-Instabilität: dort erfüllen andere Zellen den Erfolgskorridor (Längen 10.5, 12.5, 29.0) als bei Cap 45 — einzelne Instanzen verschieben die Zellauswahl, aber die grobe Größenordnung (>10, oft >15, bis in die 40er) bleibt über beide Caps stabil.

**Warum der alte Zielkorridor strukturell kaum in [8,15] zu treffen war:** Die beiden Sweep-Achsen (Rooms/Ingredients/Operations für Schwierigkeit, Episodenlänge für den nötigen Lösungsweg) sind nicht unabhängig — dieselben Parameter, die C0-Erfolg in den 30–50-%-Korridor senken (mehr Räume, mehr Zutaten, mehr Operationen), verlängern gleichzeitig den nötigen Lösungsweg. Ein starres UND aus "Erfolg 30–50 %" und "Länge 8–15" verlangt de facto eine seltene Koinzidenz zweier gekoppelter Variablen, kein unabhängig erreichbares Ziel.

**Methodische Prüfung — schadet eine längere Episode H3?** H3 ist der Signal-×-`position_norm`-Interaktionstest; die `position_norm`-Formel selbst bleibt unverändert (eingefrorene Invariante, nicht angefasst). Die eigentliche Anforderung hinter der alten 8–15-Zahl ist Positionsauflösung, nicht die Zahl selbst — vgl. Kap. 4 §4.2.2 der Thesis-Prosa, die Tower of Hanoi exakt deshalb als H3-exploratorisch statt konfirmatorisch einstuft ("whose short and near-binary temporal structure limits it as an H3 instrument"; ToH liegt laut `chapters/outline.md` bei 7–15 optimalen Steps). Mit einer Untergrenze von 8 hätte sich TextWorld mit genau dem Bereich überlappt, der für ToH bereits als für H3 unzureichend gilt — die alte Untergrenze war also eher zu niedrig als zu hoch angesetzt. Eine längere Episode erhöht `position_norm`-Auflösung monoton (feinere Schrittweite zwischen 0 und 1) und schadet H3 nicht durch die Länge allein. Die beiden real existierenden Sorgen zu langen Episoden — Solution-Space-Kompression am Episodenende (`gate_p1_readiness.md`, Akzeptiertes Restrisiko #6) und Lost-in-the-Middle unter Full-History-Prompting (Restrisiko #2, Kap. 5 §5.9 der Prosa) — sind beide bereits als akzeptierte, nicht-blockierende Limitationen dokumentiert, unabhängig von der absoluten Episodenlänge; eine Verlängerung auf bis zu 40 Steps macht diese Limitationen graduell sichtbarer, führt aber zu keinem neuen, bisher unbenannten Risiko.

**Neue Vorgabe:** Zielkorridor bleibt **30–50 % C0-Erfolg** (weiterhin HART, unverändert). Längen-Guidance wird zu **15–40 Steps** (mittlere Episodenlänge erfolgreicher Episoden) revidiert und von einem harten UND-Kriterium zu einer weichen Guidance mit einer einzigen load-bearing Grenze abgeschwächt: nur die **Untergrenze (15)** ist load-bearing für H3-Positionsauflösung (siehe Argument oben); die **Obergrenze (40)** ist ein praktischer, weicher Bezugspunkt (deutlich unter dem getesteten Production-Cap 45, verhindert Konfundierung mit Cap-nahen Trunkierungsartefakten), keine zweite harte Bedingung.

**Geänderte Dateien (synchron gehalten):**
- `blueprints/gate_p1_readiness.md` (Gate D, HART-Punkt "TextWorld-Difficulty-Sweep")
- `docs/textworld.md` (Sweep-Zielbeschreibung)
- `../metacog-thesis/chapters/05_methodology.md` §5.5.1 (löst das dortige `[PENDING: difficulty calibration]` auf)
- `../metacog-thesis/chapters/outline.md` (§5.5-Fokus-Zeile)
- `../metacog-thesis/notes/thesis_notes.md` (neuer Abschnitt "Coding-Session-Befunde (2026-07-18)")

**Offene Punkte (bewusst nicht entschieden, zur Kenntnis):**
1. `scripts/gate_d_metrics.py::LENGTH_GUIDANCE = (8, 15)` ist die Code-Konstante, die `sweep_textworld_difficulty.py`s `inside_length_guidance`-Flag und `target_window`-Feld im Sweep-JSON speist — **nicht geändert** (liegt außerhalb des für diese Korrektur beauftragten Scopes: nur Prosa/Docs/Blueprints/Log, kein Code). Muss vor dem nächsten Bestätigungssweep (n=16–20 auf dem Pod) aktualisiert werden, sonst berichtet der Sweep-Output weiterhin gegen die alte Zahl.
2. Weitere, in dieser Korrektur bewusst nicht angefasste Fundstellen derselben alten 8–15-Zahl (außerhalb des explizit beauftragten Scopes): `blueprints/thesis_design.md` (zwei Stellen, Zeilen ~147/235 — laut `CLAUDE.md` bereits als gegenüber der Kapitel-Prosa veraltet/nachrangig markiert), `docs/gate_e_h3_power_simulation.md` (nutzt 8–15 als Input-Annahme für die am 2026-07-17 bereits abgeschlossene H3-Power-Simulation — historischer Bericht, nicht rückwirkend verändert), `docs/instrument_validation_session.md` (Freeze-Notiz erwähnt 8–15 beiläufig als "§5.5-Ziel"). Falls die Power-Simulation nach dieser Korrektur wiederholt werden soll — die dortige ToH-Vergleichszahl war explizit als "optimistischer Kontextwert gegen den TextWorld-8–15-Korridor" gerahmt, und dieser Rahmen ist jetzt selbst überholt —, ist das eine Entscheidung für den User, nicht hier getroffen.
3. `../metacog-thesis/notes/thesis_notes.md`, bestehender Eintrag "Long sequential episodes: With 8–15 steps per episode…" (Prompt-Strukturierungs-Begründung, nicht Kapitel-Prosa): historischer, datierter Eintrag, bewusst nicht rückwirkend verändert (Konsistenzlog-Konvention: append-only, keine Geschichtsumschreibung); die Kernaussage (lange Episoden brauchen explizite Tag-Strukturierung) bleibt unter 15–40 Steps unverändert oder sogar verstärkt gültig.

**Nicht angefasst:** `position_norm`-Formel (eingefrorene Invariante, nicht Gegenstand dieser Korrektur); finale Korridor-Zell-Auswahl (separater, laufender n=4→n=16–20-Bestätigungssweep-Schritt, unabhängig von dieser Kriteriums-Korrektur); kein Produktionscode (`src/`) geändert.

## 2026-07-18 — Fix: C2-Tie-Break-RNG-Unabhängigkeit in `run_adaptive_episode`

**Zweck:** Behebt die im Housekeeping-Pass vom 2026-07-17 gefundene, aber bewusst zurückgestellte
offene Frage #1 (siehe Eintrag darunter): `run_adaptive_episode` (`src/agent/base_agent.py`) baute
die Step-Funktion über `resolve(stage)` bei **jedem Schritt neu** statt einmal pro Episode. Für C2
bedeutet das: `get_step_fn("C2", ...)` initialisiert `c2_call_index = 0` als lokale, per Closure
gekapselte Zählervariable — ein Neuaufbau bei jedem Step setzt diesen Zähler jedes Mal zurück, sodass
`call_index` in `c2_step_core` (`src/agent/stages/c2.py`) für **jeden** C2-Step derselben Episode
konstant 0 blieb. Da der Tie-Break-RNG (`_seeded_rng(tie_break_seed, call_index)`) über
`tie_break_seed` (konstant = `episode_id`) **und** `call_index` geseedet wird, nutzte jeder
Stimmengleichstand innerhalb derselben Episode exakt dieselbe Zufallsziehung statt unabhängiger
Ziehungen pro Step.

**Fix:** `run_adaptive_episode` cached die aufgelöste Step-Funktion jetzt pro Stage
(`step_fn_cache: dict[str, StepFn]`) für die Lebensdauer der Episode, statt sie pro Step über
`resolve(stage)` neu zu bauen. Für C0/C1 folgenlos (deren Closures sind zustandslos, geprüft am
Code); für C2 zählt `c2_call_index` jetzt korrekt über die ganze Episode hoch (`0, 1, 2, …`), sodass
jeder Tie-Break einen unabhängigen Zug bekommt. Phase 1 (`run_phase1_job`) war nie betroffen (baut
`step_fn` bereits außerhalb jeder Schleife). Kein Rückwirkungsrisiko — Phase 2 lief real noch nicht.

**Test:** neuer Regressionstest
`tests/test_base_agent_adaptive.py::test_run_adaptive_c2_call_index_increments_across_episode_steps`
(monkeypatcht `c2_step_core`, erzwingt Strategie `always_c2` über 3 Steps, prüft
`captured_indices == [0, 1, 2]`; wäre vor dem Fix `[0, 0, 0]` gewesen). Volle Suite:
**352 passed** (Baseline vor diesem Fix: 351 — siehe Housekeeping-Eintrag).

## 2026-07-17 — Codebase-Housekeeping: fünf parallele Audits (src/agent, environments/signals/execution, analysis, utils, scripts)

**Zweck:** Allgemeine Fehler-/Ungereimtheiten-Suche über die gesamte Codebasis (nicht Gate-D-spezifisch), auf Wunsch parallel zu Sport-Abwesenheit. Fünf Agents in isolierten Worktrees, je ein Modulbereich, mit der Vorgabe: offensichtliche Bugs direkt fixen + testen, alles Design-/Theorie-Relevante (insbesondere die eingefrorenen Invarianten) nur als offene Frage sammeln, nicht selbst entscheiden. Alle Fix-Commits per Cherry-Pick zusammengeführt (ein Merge-Konflikt in `textworld_env.py`-Docstring, inhaltlich sinnvoll vereint). **Volle Suite danach: 351 passed, 0 failed** (Baseline vor diesem Pass: 335).

### Gefixt (8 Commits)

- **`cluster_bootstrap` sortierte NaN/Inf-Bootstrap-Replikate mit** (`src/analysis/inference.py`) — erklärt rückwirkend die im selben Tag im Gate-E-Rehearsal als "unklares Kleine-Cluster-Phänomen" notierte Anomalie (Punktschätzer außerhalb des eigenen CI): an den echten 105004-Pilotdaten reproduziert, 535/5000 Replikate waren NaN. Jetzt werden nicht-endliche Replikate vor der Perzentil-Berechnung gefiltert, effektive Anzahl mitreportiert.
- **`holm`-Multiplizitätskorrektur fehlte der Monotonie-Schritt** (`src/analysis/inference.py`) — anti-konservativ, noch nirgends verdrahtet, aber die dokumentierte Korrektur für die H1–H4-Familien; vor erster echter Nutzung gefangen.
- **`_load_merged_config`/`_deep_merge_overlay`** (`scripts/sweep_textworld_difficulty.py`) merge nur eine Ebene tief — ein Overlay-Wert auf einer verschachtelten Ebene hätte Geschwister-Keys stillschweigend gelöscht. Aktuell latent (keine Config trifft diese Tiefe), gleiche Fehlerklasse wie die gestrigen Gate-E-Bugs. Jetzt echter rekursiver Merge.
- **YAML-Boolean-Falle in `compute_stage_selection.py`**: `compute_stages: no/yes` hätte (da `bool` in Python eine `int`-Subklasse ist) stillschweigend auf 0 bzw. `["C0"]` reduziert statt zu fehlern — dieselbe Fehlerklasse wie der `logprob_sidecar_mode: off`-Bug von heute Morgen. Jetzt expliziter Type-Guard.
- **`load_pilot_config_with_lmstudio_override` (`src/utils/pilot_config.py`) löste `extends:` gar nicht auf** — **zweite, unabhängige Instanz** desselben Bug-Musters, betrifft `run_pilot.py`, `run_c1_handoff_gate.py`, `benchmark_inference.py`. Live reproduziert: Laden von `configs/dev/gate_d_diagnostic.yaml` über diesen Pfad lieferte eine Config ganz ohne `model`-Key. Jetzt `load_yaml_with_extends()` (gleicher Algorithmus wie oben).
- **`inspect_gate_d_abort_actions.py` ignorierte den Cap des replayten Sweeps** — Copy-Paste-Divergenz zum Schwester-Script `analyze_gate_d_abort_distance.py`, das den Cap korrekt aus `sweep_results.json` liest. Ohne expliziten `--obs-ceiling` hätte ein Replay der Cap-70-Ergebnisse stillschweigend unter Cap 25 gelaufen. Jetzt Default aus dem Sweep selbst.
- Totcode-Bereinigung in `token_entropy.py` (zwei byteidentische Branches) und Docstring/Test-Lücke in `textworld_env.py` (undokumentierter `reward<0.0`-Illegal-Fallback ohne Admissible-Cache) — keine Verhaltensänderung, nur Doku + Testabdeckung.
- Stale Tuple-Arity-Docstrings in `compute_stages.py`/`base_agent.py` ("9-tuple" statt tatsächlichem 10-Tuple inkl. `call_detail`) korrigiert.
- `docs/scripts.md`: 22 Python- und 2 Shell-Scripts nachgetragen, die seit Gate-C/D-Arbeit fehlten.

### Offene Fragen für den User (bewusst nicht entschieden)

1. ~~**Echter Bug im C2-Tie-Break-RNG für Phase-2-Adaptive-Episoden**~~ — **Gefixt 2026-07-18**, siehe Eintrag oben. `run_adaptive_episode` baute die Step-Funktion bei **jedem Schritt neu** statt einmal pro Episode — dadurch setzte sich `c2_call_index` jedes Mal auf 0 zurück, und jeder C2-Tie-Break **innerhalb derselben Episode nutzte denselben RNG-Seed** statt unabhängiger Ziehungen. Betraf nachweislich `episode_runner.py::run_phase2_job` (alle adaptiven Strategien); Phase 1 war unberührt (baut die Step-Funktion einmalig). Kein Rückwirkungsrisiko (Phase 2 lief real noch nicht).
2. Toter Eps-Mismatch-Check in `ExecutionConfig.validate_frozen()` — berechnet die Abweichung von `frozen_tle_invariance_eps`, tut dann aber nichts (`pass`). Könnte vergessenes `msgs.append(...)` sein oder bewusst so (Kommentar: "frozen eps is authoritative when set", vermutlich Rückstand der Gate-C-1-N=32-Ausnahme). Sitzt auf der eingefrorenen N/eps-Invariante — nicht angefasst.
3. Totcode in einem Legacy-VC-Schwellenwert-Pfad (`thresholds.py::derive_stage_thresholds`, nur relevant wenn ein Run kein `holdout`-Feld hat — für echte Phase-1/2-Daten nicht der Fall) und eine rein deskriptive Positions-Binning-Funktion (`calibration_by_step_position`), die `step_index/episode_length` statt der eingefrorenen `position_norm`-Formel nutzt — korrekt als "nicht die konfirmatorische H3-Kovariate berührend" eingeordnet (die nutzt durchgängig `position_norm`), nur zur Kenntnis.
4. `scripts/probe_vllm_logprobs.py` und `scripts/probe_lmstudio_thinking_toggle.py` nutzen noch den ungefixten `load_yaml_path`-Pfad (nicht den jetzt gefixten `load_pilot_config_with_lmstudio_override`) — geringes aktuelles Risiko (eigenständige Pilot-Configs als Default), aber dieselbe Lücke, falls je auf ein `configs/dev/*.yaml`-Overlay gezeigt.

**Nicht gefunden (positiv zu vermerken):** keine weitere Instanz des `get_step_fn`-ohne-`resolve_step_fn_kwargs`-Wiring-Bugs über die sechs bereits gestern gefixten Scripts hinaus — systematisch an allen `get_step_fn`/`create_execution_backend`/`run_episode`-Aufrufstellen in `scripts/` geprüft.

**Hinweis:** zwei der fünf Worktrees waren versehentlich von einem alten Commit-Stand abgezweigt (vor den gestrigen Gate-D/E-Fixes) und wurden nicht vorab nachgezogen; nur ihre eigenen Fix-Commits wurden per Cherry-Pick übernommen (nicht die ganzen Branches gemerged), um nichts zurückzurollen.

## 2026-07-17 — Gate E (WEICH): H3-Power-Simulation durchgeführt

**Zweck:** `blueprints/gate_p1_readiness.md`, Gate E, WEICH-Punkt „H3-Power-Simulation" —
§5.8 sieht eine simulationsbasierte Power-Prüfung für die H3-Interaktion (Signal × `position_norm`)
vor, geseedet mit Pilot-ICC und Entropieverteilung, weil der konfirmatorische GEE-Interaktionstest
keine geschlossene Power-Formel hat und der Pilot (9–12 Cluster/Domäne) die Interaktion selbst nicht
verlässlich schätzen kann. Der WEICH-Punkt verlangt „durchführen oder die Limitation aktiv wählen" —
dieser Eintrag dokumentiert die Durchführung.

**Methode:** `scripts/h3_power_simulation.py` (neu). ICC (zwei Methoden: GEE-`dep_params` primär,
ANOVA-ICC(1) als Kreuzcheck) und TLE/VC-Verteilung aus den 72 echten Pilot-Episoden
(`data/results/instrument_validation/phase1_20260714_105004/`, dieselbe Quelle wie
`docs/gate_e_rehearsal.md`). Echte Monte-Carlo-Simulation (kein geschlossenes Formel-Substitut):
geclusterte Random-Intercept-Logit-Datensätze unter dem geplanten Phase-1-Design (50 Instanzen/
Domäne, 5 Runs × 3 Stages gepoolt = 750 Episoden/Domäne) über ein Raster wahrer
Interaktionseffektgrößen simuliert, jeder Datensatz mit der **echten** Produktionsfunktion
`src/analysis/inference.py::fit_h3_model` (GEE, Exchangeable, Binomial) gefittet, empirische
Ablehnrate = Power. 400 Replikate/Zelle (TLE-Hauptraster), 250 (VC-Sekundärraster),
Multiprocessing über 7 Worker. Typ-I-Fehler-Check bei wahrem Nulleffekt bestätigt die
Simulationsmaschinerie (Ablehnrate nahe/leicht unter dem nominalen Alpha in allen vier Zellen).

**Ergebnis:** siehe `docs/gate_e_h3_power_simulation.md` für den vollständigen Report
(Pilot-ICC/Entropie-Anker, Power-Tabellen, 80-%-Kreuzungspunkte, Annahmen). Kernbefund: der
konfirmatorische TextWorld/TLE-H3-Test erreicht 80 % Power nur für Interaktionseffekte in der
Größenordnung des vollen Haupteffekts (praktisch vollständiger Signalverlust bis Episodenende,
$\beta_{int}\approx-0.31$ bei $\beta_z=0.30$); für plausiblere moderate Degradationsgrade
(25–50 % Abschwächung, $\beta_{int}\approx-0.075$ bis $-0.15$) liegt die Power nur bei ~15–35 %.
Tower of Hanoi zeigt in der Simulation höhere Power (niedrigere ICC, größerer Haupteffekt-Anker),
aber mit einer wichtigen Einschränkung: die ToH-Episodenlängen stammen mangels eigenem
Längenkorridor aus den noch nicht Gate-D-kalibrierten (nahe-Cap) Pilotlängen, während TextWorld
den bereits definierten 8–15-Step-Zielkorridor nutzt — die ToH-Zahl ist daher ein optimistischer
Kontextwert, nicht die belastbarste Aussage; ToH ist ohnehin nur exploratorisch (kein
Holm-Familienschutz).

**Einordnung:** kein HART-Blocker, keine Design-Änderung an diesem Punkt vorgenommen — die
Simulation liefert eine quantifizierte Grundlage für die Interpretation eines nicht-signifikanten
H3-Ergebnisses in Kapitel 6/7 (Nullbefund ≠ „keine Degradation", sondern „große Degradation
ausgeschlossen, moderate bleibt plausibel und unterpowert"), konsistent mit der bereits in §5.9
benannten Sorge zu Interaktionstest-Power. `blueprints/gate_p1_readiness.md`s WEICH-Checkbox-Zeile
trägt jetzt einen Verweis auf diesen Eintrag; die Checkbox selbst bleibt unangetastet (offen), da
keine Gate-Entscheidung getroffen wurde.

**Testsuite:** keine Quelländerung außerhalb von `scripts/` und `docs/`; `python -m pytest tests/ -q`
nicht erneut nötig, da kein Produktionscode berührt wurde (nur ein neues, eigenständiges Analyse-
Skript, das ausschließlich bestehende, bereits getestete Funktionen (`load_run_dataset`,
`fit_h3_model`) aufruft).

## 2026-07-17 — C2-Tie-Break quantitativ geprüft: User-Hypothese (Temperatur-Diversität schlägt einzelne korrekte Kandidaten) selten, dominanter Effekt liegt vor der Abstimmung

**Zweck:** Quantitative Nachprüfung der gestrigen qualitativen Beobachtung ("in den Traces wählte
Voting teils einstimmig eine geteilt-falsche Aktion", Eintrag "Korridor-Kriterium korrigiert" oben)
und einer eigenständigen Nutzerhypothese: schlägt C2s Sampling-Temperatur gezielt einen einzeln
korrekten Kandidaten gegen zwei unabhängig-falsche zusammenlaufende Kandidaten? Bisher nur an n=2
Episoden eyeballed, nicht gemessen.

**Methode (Replay, kein Proxy):** Für beide verfügbaren C2-TextWorld-Traces mit vollem
Per-Step-Log (`trace_trace_tw_C2_{0,1}.jsonl` — es existieren nur diese zwei aus der gestrigen
`gate_d_trace_probe.py`-Session, keine weiteren) wurden die exakt gleichen TextWorld-Spiele
deterministisch neu generiert (`_generate_combo_games` mit identischem Seed/Params) und
byte-identisch gegen die gespeicherte `reset_observation` verifiziert. Anschließend Replay der
**tatsächlich committeten** Aktionsfolge durch `TextWorldEnv.step()` — Korrektheitslabel und
Quest-Distanzen stimmen für **alle 27 Steps** (19+8) exakt mit den gespeicherten
`step_correctness`-Einträgen überein (kein einziger Abweicher), bevor der Ansatz auf die
**Kandidaten**-Branches angewendet wurde: pro Step wird der Replay bis zum Pre-Step-Zustand
wiederholt und für jeden der 3 rohen `subcalls[i].action_exec`-Kandidaten separat ein
`env.step(candidate)` ausgeführt — dieselbe `_classify_quest_correctness`/Illegal-Logik, die die
Produktions-Pipeline für die committete Aktion nutzt, jetzt pro Kandidat statt nur für den Gewinner.
Kein Proxy-Fallback nötig. Wegwerf-Skript unter `/tmp` (nicht committed, kein Produktionscode
berührt).

**Ergebnis (n=27 Steps, 81 Kandidatenaktionen, 2 Episoden — explizit klein):**

| Kennzahl | Wert |
|---|---|
| Kandidatenlabel-Verteilung (81 Aktionen) | illegal 37,0 %, legal 29,6 %, **optimal 19,8 %**, unlabeled 8,6 %, leer/unparsebar 4,9 % |
| Steps mit 0 korrekten Kandidaten (von 3) | **20/27 (74 %)** |
| Steps mit genau 1 korrektem Kandidaten (User-Szenario) | **1/27 (3,7 %)** |
| — davon: einzeln-korrekter Kandidat verliert die Wahl/Tie-Break | **1/1** (einziges Vorkommen) |
| Steps mit ≥2 korrekten Kandidaten | 6/27 — Gewinner in **6/6** korrekt |
| `tie_broken=True` (RNG-Tie-Break aktiv) | 10/27 (37 %) — deckt sich mit der gestrigen "7/19"-Beobachtung (ep0: exakt 7/19) |
| — davon Tie-Break landet auf korrektem Kandidaten | **0/10** |

**Befund:** Der Tie-Break selbst hat in dieser Stichprobe **nie** einen korrekten Kandidaten
verworfen — in allen 10 Tie-Break-Fällen war bereits keiner der 3 Samples optimal, die RNG
arbitriert also unter bereits gleichwertig-suboptimalen Exploration-Aktionen (`examine`, `look`,
`go west` an nicht-quest-relevanten Abzweigungen), nicht zwischen richtig und falsch. Die
Nutzerhypothese (temperaturgetriebene Diversität lässt eine korrekte Probe gegen zwei unabhängig
falsche verlieren) trat **genau einmal** auf (`trace_tw_C2_0`, Step 11: `go east` optimal vs. `go
north`×2 legal, 2:1-Mehrheit gegen die korrekte Aktion) — real, aber mit n=1 nicht verallgemeinerbar.
Der weit dominantere Effekt (74 % aller Steps) ist, dass **keiner** der 3 C2-Samples überhaupt eine
optimale Aktion trifft — das Problem liegt vor der Abstimmung, nicht in ihr.

**Gegenprobe zur "gemeinsamer Konsens beendet Episode vorzeitig"-Theorie:** Beide getraceten
C2-Verluste enden an Steps mit game-endendem `lost=True` (`trace_tw_C2_1` Step 7: einstimmig 3/3
`cook yellow onion with stove`, unlabeled/lost). Der C1-Sibling-Trace **auf demselben Game-Instance**
(`trace_tw_C1_1`, ein einzelner Sample, keine Abstimmung) trifft an **exakt demselben Step dieselbe
verlierende Aktion** — die Ursache ist ein geteilter Modell-/Parser-Kompetenzengpass (Onion nie
erfolgreich genommen/geschnitten über 6 vorangehende Illegal-Versuche), keine C2-spezifische
Konsens-Verstärkung durch das Voting selbst.

**Fazit:** Weder die Tie-Break-Hypothese noch die Konsens-Termination-Hypothese sind in dieser
Stichprobe die primäre Erklärung für C2 < C1 bei TextWorld — beide sind reale, aber seltene/geteilte
Randeffekte. Der Haupttreiber scheint eine Generierungs-Abdeckungslücke zu sein (die optimale Aktion
wird in 74 % der Steps in keiner der 3 Samples überhaupt vorgeschlagen), gemeinsam mit C1. **Kein
Signifikanztest möglich** — n=2 Episoden, n=1 Zielereignis. Für echte Konfidenz bräuchte es einen
größeren C2-Trace-Sweep (Größenordnung 25–40 Episoden / 500+ Steps, um bei der beobachteten
~4-%-Basisrate auf ≥20 Zielereignisse zu kommen) sowie gepaarte C1-Traces auf denselben
Game-Instanzen, um den geteilten-Kompetenz-Confound sauber zu trennen.

**Nicht angefasst:** Kein Produktionscode geändert (reine Read-only-Analyse bestehender Traces).
Keine Gate-D/E-Statusänderung — dies ist eine Verfeinerung einer bereits dokumentierten
Beobachtung, kein neuer Blocker.

## 2026-07-17 — Rückwirkungs-Check: Gate-E-Bugfixes betreffen keine bereits berichteten Gate-D/C-Ergebnisse

**Zweck:** Die drei im Gate-E-Rehearsal gefundenen Bugs (siehe Eintrag unten) verifiziert gegen
"hätte das ein vorher als sauber berichtetes Ergebnis verfälscht?" — per Grep auf tatsächliche
Aufrufer, nicht nur Vermutung.

- **`extends`-Overlay-Bug** (`run_phase1.py`/`run_phase2.py`): kein einziges Gate-D-Sweep-/
  Feasibility-Script importiert diese Funktion — alle nutzen die korrekte `_load_merged_config` aus
  `sweep_textworld_difficulty.py`. Unberührt.
- **`logprob_sidecar_mode: off`-Crash** (`LogprobSidecarConfig.from_logging_config`): Aufrufer sind
  ausschließlich `episode_runner.py`, `run_phase1.py`, `run_phase2.py`, `smoke_parallel.py` —
  keines davon lief gestern. Wäre zudem ein **Crash**, kein stiller Fehler — nichts ist gestern
  gecrasht.
- **`load_run_dataset` verwirft Phase-2-Episoden:** betrifft nur episodenweise `strategy`-Daten;
  solche existierten vor dem heutigen Gate-E-Rehearsal nicht (Phase 2 läuft real noch nicht). Die
  Gate-C-Nutzung von `diagnose_tle_distribution.py` auf `105004` ist Phase-1-förmig
  (`compute_stage`-Feld vorhanden) und damit ebenfalls unberührt.

**Fazit:** Alle drei Bugs sind in genau der Session aufgetreten, die sie auch gefangen hat — keine
Korrektur an früher berichteten Zahlen nötig.

## 2026-07-17 — Gate E: Analyse-Rehearsal (End-to-End-Trockenlauf) auf Gate-C-Pilotdaten

**Zweck:** Der zweite HART-Punkt aus Gate E — kompletter Trockenlauf der konfirmatorischen Kette
Episode-JSONs → Step-Tabelle → `grid_search_thresholds` → Policy-Artefakt → `load_policy` →
`run_phase2.py`-Smoke (`adaptive_tle`, Mock) → `cluster_bootstrap` auf ΔAUROC(TLE, VC) — auf den 72
echten Gate-C-Episoden (`data/results/instrument_validation/phase1_20260714_105004/`), **vor**
echten Phase-1-Daten. Voller Bericht mit Zahlen: [`docs/gate_e_rehearsal.md`](gate_e_rehearsal.md).

| Bereich | Ergebnis |
|---------|----------|
| Step-Tabelle (`datasets.py`) | **OK** — 72 Episoden → 1363 Steps, 0 fehlende Spalten |
| Künstlicher Holdout-Split | **Workaround** — instance<3/12 je Domäne (25 %, nicht die realen 5/50); begründet in `gate_e_rehearsal.md` §1 |
| Grid-Search + Policy-Artefakt | **OK** — `objective_definition=step_level_proxy_v1`, `theta1=0.8`/`theta2=0.9` in allen 4 Domäne×Signal-Zellen (Extremwert-Beobachtung notiert, kein Bug) |
| `load_policy` Sanity | **OK** — stage-wise ECDF lädt, low/mid/high-Probe → C0/C0/C2 in beiden Domänen |
| `run_phase2.py`-Mock-Smoke | **OK** — 12/12 Episoden, 0 Fehler, Policy-Artefakt-SHA-256 in `run_metadata.json` |
| `cluster_bootstrap`(ΔAUROC) | **OK** — gepoolt point=0.101, 90 %-CI [0.047, 0.157], n=1018 Steps/18 Cluster |
| **Fund 1:** `run_phase1.py`/`run_phase2.py::load_config` ignorierte `extends` | **Fixed** — beide nutzen jetzt `scripts.sweep_textworld_difficulty._load_merged_config` (wie die vier Gate-D-Diagnoseskripte bereits) |
| **Fund 2:** YAML-Falle `logprob_sidecar_mode: off` → Python-Bool `False` → `ValueError` | **Fixed** — `_normalize_mode` (`src/utils/logprob_sidecar.py`) behandelt `False` explizit als `"off"`; betraf 6 bestehende Dev-Configs, bisher nie ausgelöst |
| **Fund 3:** `load_run_dataset` verwarf jede Phase-2-Episode still (kein `compute_stage`) | **Fixed** — `_validate_episode_record` akzeptiert `compute_stage` **oder** `strategy`; Regressionstest ergänzt |
| Testsuite lokal | **OK** — `python -m pytest tests/ -q` → **335 passed**, 0 skipped |

**Nicht angefasst:** `blueprints/gate_p1_readiness.md` Gate-E-Checkbox (Status-Entscheidung bleibt beim
Nutzer, beide Gate-E-HART-Punkte — Pre-Analysis-Screen separat, dieses Rehearsal hier — sind jetzt mit
Evidenz belegt).

## 2026-07-17 — ToH-Repräsentationskonvention: Bottom-to-Top-Leserichtung der Peg-Listen expliziert

**Problem:** Die ToH-Beobachtung zeigt den Zustand als `Peg A: [3, 2, 1]`, sagte aber nie, **welches
Listenende die "oberste" Scheibe** ist (die einzige bewegbare). Der Code behandelt das *letzte*
Listenelement als top (`state[src][-1]` in `_legal_moves`/`_apply_move`), diese Konvention wurde dem
Modell jedoch nie genannt. Ohne sie kann ein kleines Modell die eigene Zuglegalität nicht
selbstprüfen (Größenregel), was den bekannten C0-Nullbefund (Illegal-Rate 75–89 %, wiederholtes
Anbieten desselben illegalen Zugs gegen unveränderten Zustand — Diagnose 2026-07-16) mit-erklärt.

**Fix:** Eine reine Repräsentationsaussage in `TowerOfHanoiEnv._render_observation`
(`src/environments/tower_of_hanoi.py`), direkt unter den drei Peg-Zeilen:
`"Each peg's disks are listed bottom-to-top, so the last (rightmost) number is the top disk."`
Die Renderfunktion ist die **einzige** Stelle, an der die Listendarstellung erscheint, und wird von
`reset()` **und** `step()` geteilt → Konvention erscheint automatisch in jeder Beobachtung über
C0/C1/C2 (kein zell-/schwierigkeits-/stufenspezifischer Prompt-Variant). Kein Eingriff in
`domain_prompts.tower_of_hanoi.prefix`; der Prefix zeigt die Liste gar nicht, dort fehlt der Anker.
`configs/dev/gate_d_calibration.yaml` und `gate_d_diagnostic.yaml` überschreiben `domain_prompts`
nicht (nur `extends`), erben den Fix also identisch.

**Verortung (analog TextWorld-Vokabular-Fix 2026-07-15, gleiches DV-Schutz-Prinzip):** TextWorld
editierte den Config-`prefix`, weil dort die Beobachtung roher Engine-Text ist, den wir nicht
autoren — der Prefix ist die einzige autorisierbare Fläche. ToH autort die Beobachtung selbst und
dupliziert Regeln/Ausgabeformat bereits in `_render_observation`; die konsistente Stelle für die
Leserichtung ist daher die Renderfunktion, direkt neben den Daten.

**Abgrenzung (DV-Schutz):**
- Reine Repräsentationsaussage (**wie** die Liste zu lesen ist), **kein** Zugbeispiel. Das bestehende
  Ausgabeformat-Beispiel `e.g. A->C` (nur Syntax `[source]->[target]`) bleibt unverändert; es wurde
  bewusst **kein** neues Peg-zu-Peg-Zugbeispiel ergänzt (Priming-Risiko auf gezeigte Pegs/Richtung).
- Nennt **keine** legalen/admissiblen Züge, keine Strategie, keinen Zielpfad. `include_valid_moves`
  bleibt `false` und unberührt; die Zugauswahl bleibt vollständig generativ.
- Verbindet lediglich die bereits vorhandene Regel "move only the top disk" mit der Frage, welche
  Scheibe das ist — ermöglicht Selbstprüfung der Legalität, nicht die Zugwahl.

**Bewusst nicht geändert:** `domain_prompts.tower_of_hanoi.prefix` (kein Listen-Anker),
Parsing (`_parse_action`), Label-Logik (`correctness`), TextWorld (unberührt).

**Test:** `tests/test_07_tower_of_hanoi.py` — neue Fälle
`test_env_reset_states_bottom_to_top_convention`, `test_env_step_observation_repeats_convention`
(Konvention in Reset- **und** Step-Beobachtung) sowie erweiterter
`test_env_reset_can_include_valid_moves_opt_in` (Insert-Index-Shift geprüft: `Valid moves:` bleibt
vor der Reply-Format-Zeile). **Volle Suite: 332 passed** (`python -m pytest tests/ -q`).

**Kein Empirie-Lauf:** Strukturell/Unit-getestet; die Verhaltenswirkung auf den C0-Nullbefund
(Pod/vLLM) ist bewusst ein separater Folge-Schritt, nicht Teil dieser Änderung.

## 2026-07-17 — Korridor-Kriterium korrigiert: kein striktes C0<C1<C2-Gefälle verlangen

**Korrektur zu:** Dem am 2026-07-16 vorgeschlagenen Zwei-Teil-Korridor-Kriterium (Opus-Analyse).
Teil (b) forderte ein "nachgewiesenes C0<C1<C2-Erfolgsgefälle" — das ist durch die eigenen Befunde
widerlegt: TextWorld zeigt C1 (100%) > C2 (50%) in `r3_i1_take-only`, ToH zeigt C1=C2 (62,5%
gleichauf). C2 schlägt C1 in keiner bisher getesteten Zelle. **Korrigiertes Kriterium:** "C0
deutlich unter {C1, C2}", ohne strikte Reihenfolge zwischen C1 und C2 zu verlangen.

**Bezug:** Der C1>C2-Befund ist theoretisch bereits durch Koriats Self-Consistency-Modell der
Konfidenz und die zitierte Uncertainty-Collapse-Literatur (Kap. 2 der Thesis) erklärbar — Konsens
unter Selbstkonsistenz-Sampling spiegelt Übereinstimmung, nicht Korrektheit; in den Traces wählte
Voting teils einstimmig eine geteilt-falsche Aktion. **Entschieden (2026-07-17, User): 3
Compute-Stufen bleiben** (Option B statt Reduktion auf 2) — Reduktion wäre unter Zeitdruck netto
mehr Schreibarbeit gewesen (betrifft H2/Always-C2-Baseline, Baseline-Tabelle, C2-Methodik). Nur noch
Prosa-Umsetzung offen (Kap. 5 Tab. 5.1 "Upper Bound" → "höchste Compute-Stufe" abschwächen, C1>C2
theoretisch verankern) — Details in `../metacog-thesis/notes/thesis_notes.md`
("Coding-Session-Befunde (2026-07-17)"). Keine weitere Code-Entscheidung nötig.

## 2026-07-16 — ToH C0/C1/C2-Feasibility: C0-Nullbefund löst sich unter Reasoning auf

**Zweck:** Direkte Anschlussfrage an den ToH-C0-Nullbefund (siehe Eintrag unten): löst sich das durch
Reasoning auf, wie bei TextWorld? Kein bestehendes Script deckte ToH bei C1/C2 ab
(`sweep_toh_difficulty.py` ist C0-only); dafür Einweg-Script `toh_feasibility.py` (nicht ins Repo
committed, nur der Ergebnis-JSON) gebaut: 8× 3-Disk-Instanzen (seed 42, `partial_start_range=(0,3)`,
`max_steps` pro Instanz = `optimal_steps*3`), `configs/dev/gate_d_calibration.yaml`,
`resolve_step_fn_kwargs` korrekt verdrahtet.

`data/results/gate_d_calibration/toh_feasibility/toh_feasibility_report.json`:

| Stage | Erfolg |
|---|---|
| C0 | 0/8 = 0,0 % |
| C1 | 5/8 = 62,5 % |
| C2 | 5/8 = 62,5 % |

**Befund:** Der C0-Nullbefund ist kein Bug und keine unlösbare Aufgabe — er ist reine
C0-Schwäche (kein Reasoning). Mit Reasoning (C1) springt der Erfolg auf 62,5 %, weit über den
30–50-%-Zielkorridor. **Anders als bei TextWorld liegen C1 und C2 hier gleichauf** (kein C1>C2-Gap;
vgl. TW-Feasibility unten: C1 100 %, C2 50 %). Alle C0-Episoden liefen exakt bis zum jeweiligen
Instanz-Cap durch, ohne zu gewinnen (`episode_length_steps == max_steps` in allen 8 Fällen) — passt
zur schon bekannten hohen Illegal-Rate (75–89 %) aus dem C0-Sweep: die Aktionswahl selbst ist ohne
Deliberation kaum brauchbar, nicht die Aufgabenschwierigkeit an sich.

**Konsequenz für Gate D:** ToH-Schwierigkeitskalibrierung braucht vermutlich C1 oder C2 als
Referenzstufe, nicht C0 — 3 Disks bei C0 ist strukturell zu hart für den 30–50-%-Korridor. Das ist
eine Design-relevante Beobachtung (keine autonome Entscheidung getroffen, nur dokumentiert).

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
