# GATE P1 — Phase-1-Readiness (Go/No-Go vor der Kalibrierungs-Datenerhebung)

Stand: 2026-07-06 (rev. 2). Zweck: Ein einziges Dokument, das entscheidet, ob die Phase-1-Datenerhebung starten darf. Die Code-Basis ist nach PR0–PR2 und Gate A/B designkonform und eingefroren. Diese Revision zieht zwei operative Entscheidungen nach: **Produktions-GPU ist die RunPod RTX 5090 (32 GB)**, und **Inferenz läuft über ein parallelisiertes vLLM-Backend (Server mit Continuous Batching)** statt sequenzieller In-Process-Aufrufe. Beides ändert nicht die Messmethodik, aber die Verifikations- und Budget-Punkte.

**Entscheidungsregel:** Phase 1 startet, wenn alle Punkte in Gate A bis F mit Status HART erfüllt sind. WEICH-Punkte dürfen offen sein, werden aber vor dem Start explizit als akzeptiertes Risiko in dieses Dokument eingetragen (Datum + Begründung). Jeder HART-Punkt produziert ein archiviertes Evidenz-Artefakt.

**Statusübersicht:** Gate A abgeschlossen (Commit `c0d4067`, Prosa-Freeze Kap. 5). Gate B abgeschlossen (Merge `6ca2857`, Suite lokal + Pod grün, Persistenz- und Roundtrip-Nachweis). Gate C abgeschlossen (Merge `9f8dafd`, C-0…C-6 auf Pod 5090; 317 pytest grün). Offen: Gate D, E, F.

---

## Vorbemerkung — ein tragender Frühbefund (nicht überlesen)

Die in Gate B eingecheckte reale Smoke-Episode (`tests/fixtures/episode_compact_real.json`, Qwen3-4B) zeigt ein Muster, das Gate C zwingend auf dem Produktionsmodell ausräumen muss: TLE-Mean-Entropy durchgängig in der Größenordnung 1e-4 bit, VC durchgängig `null`, alle Steps `illegal`. Auf dem 4B-Smoke mit wenigen Schritten ist das kein Beweis gegen das 8B-Modell, aber es ist exakt das Degeneriertheitsmuster, das H1a (TLE-Diskrimination) und die VC-Vergleichbarkeit unmöglich machen würde, falls es auf Qwen3-8B fortbesteht. Die beiden Gate-C-Prüfungen **C-4 (VC-Screen)** und **C-5 (TLE-AUROC-Smoke)** sind daher die eigentlichen Kipp-Punkte des gesamten Vorhabens, nicht Routine. Wenn sie negativ ausfallen, ist das ein echtes Stopp-Signal vor dem Commit der vollen Phase-1-Stunden, kein zu überschreibendes Häkchen.

---

## Gate A — Preregistrierungs-Freeze (Prosa) — ABGESCHLOSSEN

Alle sechs HART-Punkte in Kapitel 5 eingearbeitet und unter Commit `c0d4067` dokumentiert (Evidenz in `notes/thesis_notes.md`, Abschnitt „Gate A — Preregistration Freeze"): `position_norm`-Konvention, §5.4-Allocator-Nachträge (`step_level_proxy_v1`, Erste-Schritt-Regel, ECDF-Deploy-Parität), §5.2.2 judged context + Reparse, stufenspezifische Temperaturen mit `raw_logprobs`, §5.9-Ignorierbarkeit plus Step-Level-Proxy-Limitation. Der finale Freeze-Tag wird erst nach Gate F gesetzt.

---

## Gate B — Code-Readiness — ABGESCHLOSSEN

Merge `6ca2857` auf `main`. Suite lokal und auf dem Pod grün (250 passed / 0 skipped), `steps_detail`-Persistenz auf echtem Smoke belegt, Policy-Artefakt-Roundtrip auf generiertem Holdout-Artefakt mit `objective_definition=step_level_proxy_v1` bestätigt, `nearest_position`-Tie-Break und Token-Accounting-Tests grün. Evidenz in `docs/consistency_log.md`, Eintrag 2026-07-06.

---

## Gate C — Instrument-Validierung auf dem Produktions-Backend (RunPod RTX 5090, Qwen3-8B, parallelisiertes vLLM)

Begründung: Alle Lokalpilot-Evidenz (Qwen3-4B, MLX/LM Studio) und der Gate-B-Smoke belegen die Pipeline, nicht das Messinstrument. Jeder Punkt hier muss auf der Konfiguration laufen, die Phase 1 tatsächlich verwendet: 5090, `experiment_core.yaml` (nicht `pilot.yaml`), parallelisiertes vLLM-Backend. Die Reihenfolge staffelt bewusst nach steigenden Kosten mit harten Abbruchpunkten vor den teuren Läufen.

### C-0 — Umgebungs-Fixierung (kein Signal, Minuten)
- [x] **HART.** Pod (5090) hoch, Qwen3-8B auf Network Volume; Config = `experiment_core.yaml`; `logprobs_mode="raw_logprobs"` auf der Engine gepinnt; `dtype=float16` (nicht `fp16`); History-Guard aktiv (kein `--allow-history-truncation`); parallelisiertes vLLM-Backend gestartet und erreichbar; `hf_model_card_gate.py` als Read-Only-Check, der Qwen3-8B nicht über die Thinking-Heuristik flaggt (Dense-vs-MoE ist das Kriterium, Thinking ist für C1 erforderlich). Kein Abbruchkriterium außer: Modell lädt, Backend antwortet, Config ist die richtige. *Evidenz: `run_metadata` des C-0-Checks.* (2026-07-12 Pod-Session)

### C-1 — Backend-Parität, Kriterien 1 + 2 + Batch-Invarianz (billig, HARTES STOPP-SIGNAL)
- [x] **HART.** `verify_backend_parity.py --backend server` auf dem Pod. Drei entkoppelte Kriterien, jeweils separates Pass/Fail:
  1. **K-Coverage** ≥ 20 Top-Logprobs an den Action-Token-Positionen.
  2. **Temperatur-Invarianz** der TLE: identische renormalisierte Entropie bei T=0.3 vs. T=1.0 auf dem festen Probe-Set (belegt die Skalen-Invarianz aus §5.6, §5.2.1).
  3. **Batch-Invarianz (neu, wegen Parallelisierung):** derselbe Probe-Prompt liefert innerhalb der preregistrierten Toleranz (§5.7-Schwelle, `TLE_INVARIANCE_EPS_BITS`) identische Top-Logprobs, ob allein oder in einem gemischten Batch verarbeitet. Grund: Das parallelisierte Backend batcht Sequenzen nebenläufig; fp16-Reduktions-Nichtdeterminismus über wechselnde Batch-Zusammensetzungen darf die TLE-Messung einer Aktion nicht verschieben. Ohne diesen Nachweis ist die Cross-Episode-Vergleichbarkeit von TLE unter Batching nicht gesichert.
  - **Stopp:** Fällt eines der drei Kriterien, bricht Gate C hier ab, bevor eine echte Episode läuft. Batch-Invarianz-Fail heißt entweder deterministische Batching-Einstellungen erzwingen (z. B. feste Batch-Grenzen, `enforce_eager`, Seed-Pinning) oder die Toleranz empirisch als Rausch-Floor neu begründen und preregistrieren. *Evidenz: `data/results/backend_parity_<UTC>.json`, referenziert in §5.7.5.*
  - **2026-07-14:** PASS @ N=32, eps=0.05 (committed-action probes); `backend_parity_20260714T104959Z.json`; siehe Consistency-Log.

### C-2 — Format-Compliance-Probe, klein (billig-mittel, HARTES STOPP-SIGNAL)
- [x] **HART.** Wenige Episoden pro Domäne, nur Mechanik, kein Signal: C1 erzeugt einen Think-Block und eine als erste nichtleere Zeile nach `</think>` parsebare Aktion; keine Reasoning-Leakage in den Action-Slot; VC-Follow-up liefert überhaupt einen parsebaren Integer. Deckt die aus dem 4B-Pilot bekannten Fehlerbilder (leere Aktionen, Prompt-Echo, VC-immer-null, Thinking flutet Aktion) auf dem 8B ab. **Stopp:** Kein sauberes C1-Parsing → zuerst Config reparieren (`chat_template: true`, `action_stop`, `followup_max_tokens` auf den Nicht-Pilot-Wert), bevor der teure Signal-Smoke läuft. *Evidenz: Probe-Report.*

### C-3 — ToH-Parseability (billig, HARTES STOPP-SIGNAL)
- [x] **HART.** ToH-Parseability > 80 % mit Qwen3-8B bei C0 (20 Episoden, 3 Disks). Unterschreitung → `include_valid_moves`-Fallback aktivieren und als preregistrierte Konfigurationsänderung dokumentieren. *Evidenz: Pilot-Summary.* (2026-07-14: parse_rate=1.0, `toh_parse_probe`)

### C-4 — VC-Validitäts-Screen auf finaler Elicitation-Config (mittel, KIPP-PUNKT)
- [x] **HART.** Auf Qwen3-8B mit finaler Config (probscore-Prompt, `judged_context=action_only`, Retry bei T=0, `followup_max_tokens` auf Nicht-Pilot-Wert): Parse-Rate ≥ 90 % und Varianz-Screen aus `preanalysis_screen.py` (Anteil modaler Wert, Wertespektrum). Angesichts des `null`-VC-Musters der Smoke-Fixture ist dies ein realer Kipp-Punkt. Degeneriertheit ist per §5.2.2 ein berichtbarer Befund und kein automatischer Startblocker, aber der Screen muss **vor** dem Freeze laufen, damit die Elicitation nicht nach Datensichtung angepasst wird. *Evidenz: Screen-Report.*

### C-5 — TLE-Signal-Smoke, AUROC (teuer, HERZSTÜCK, KIPP-PUNKT)
- [x] **HART.** Der eigentliche Lauf: genug Episoden über C0/C1/C2 in beiden Domänen, um in einem Rutsch zu liefern: TLE-Step-Level-AUROC gegen `y_optimal` (Ziel > 0.6 in mindestens einer Domäne), die Logprob-Sidecars für C-6, den Durchsatz für Gate F und eine Difficulty-Verteilungs-Vorschau für Gate D. **Kipp-Punkt:** AUROC < 0.6 in beiden Domänen ist kein automatisches No-Go, erzwingt aber die Aggregations-/Prompt-/K-Prüfung vor dem Commit der vollen Phase-1-Stunden. Direkter Bezug zur Smoke-Fixture: liegt TLE auf dem 8B ähnlich flach (~1e-4 bit) und ohne Trennung zwischen optimal und illegal, ist das das Frühwarnsignal, das hier sichtbar werden muss, nicht erst nach der vollen Erhebung. *Evidenz: Signal-/Kalibrierungs-Summary.*

### C-6 — K-Sensitivitätssweep (billig, aus C-5-Daten, kein neuer Lauf)
- [x] **HART.** `sweep_topk_sensitivity.py` rekonstruiert TLE für K ∈ {5, 10, 20} aus den C-5-Sidecars und rechnet H1a-AUROC pro K und Domäne. §5.2.1 verspricht diesen Sweep ausdrücklich vor der Hauptdatenerhebung; K wird danach eingefroren. Kommt nach C-5, weil es dessen Sidecars konsumiert. *Evidenz: Sweep-JSON + Einzeiler im Consistency-Log.*

**Effizienz:** C-0 bis C-6 in einer Pod-Session; die harten Stopps (C-1, C-2, C-3) sind echte Abbruchpunkte, nicht nachträgliche Häkchen. Der teure Lauf ist ausschließlich C-5.

---

## Gate D — Schwierigkeitskalibrierung und Manifeste (Immutabilität)

- [x] **HART — TextWorld-Difficulty-Sweep.** Zielkorridor 30–50 % C0-Episodenerfolg auf Qwen3-8B (primäres, hartes Kriterium). **Längen-Guidance revidiert 2026-07-18** (siehe `docs/consistency_log.md`): mittlere Episodenlänge erfolgreicher Episoden **15–40 Steps**, ersetzt die ursprüngliche A-priori-Vorgabe 8–15 Steps, die durch die realen Post-Fix-Sweeps (`data/results/gate_d_calibration/textworld_sweep{,_cap70}/sweep_results.json`) empirisch kaum erreichbar war (Zellen im Erfolgskorridor liegen bei Sieg-Episodenlängen von 10.5–41.5 Steps, nicht 8–15 — die beiden Sweep-Achsen Schwierigkeit und Episodenlänge sind gekoppelt, nicht unabhängig steuerbar). Load-bearing bleibt nur die **Untergrenze** (15): H3 ist in TextWorld konfirmatorisch und braucht genug `position_norm`-Auflösung, um nicht in die für Tower of Hanoi bereits als unzureichend eingestufte Nähe zu rutschen (ToH: 7–15 optimale Steps, dort H3-exploratorisch statt konfirmatorisch). Die Obergrenze (40) ist ein weicher, praktischer Bezugspunkt (deutlich unter dem getesteten Production-Cap 45) und **kein** zweites hartes UND-Kriterium. **Korridor bestätigt** (`r5_i1_take+cook`, n=16, unabhängiger Seed 9001 gegenüber Manifest-Seed 20260718, identische Generierungsparameter): 43.75 % Erfolg, sauber im 30–50-%-Korridor — dieselbe Evidenzqualität (Validierungssample statt Lauf auf den exakten finalen 50 Instanzen), mit der die ToH-Zeile unten bereits abgehakt wurde; Nachziehen 2026-07-20. *Evidenz: `data/results/gate_d_calibration/textworld_candidate_confirmation/validation_results.json`; Korrektur-Herleitung in `docs/consistency_log.md`, 2026-07-18.*
- [x] **HART — ToH-Konfiguration (Referenzstufe revidiert 2026-07-19, siehe `docs/consistency_log.md`).** C0 erwies sich über mehrere unabhängige, echte Sweeps (n=10/20/30, gemischt und isoliert) konsistent als strukturell nahe 0 % — kein Schwierigkeitsproblem, sondern ein systematischer Bias (das Modell wählt bei C0 nie den als Ziel benannten Peg als Zugquelle, unabhängig vom Zustand; per Trace-Analyse belegt). Korridor daher an **C1** kalibriert, nicht C0 (analog zur Erfolgsrate-Logik bei TextWorld, nur mit anderer Referenzstufe). Finale Konfiguration: **4 Disks, `partial_start_mode=random_scramble`** (zieht ohne Zurücklegen aus dem vollen 3⁴=81-Zustände-Raum, statt aus den nur `num_disks+1` Zuständen des ursprünglichen deterministischen Optimalpfad-Präfixes — dieser Deckel hätte die "50 Instanzen" auf effektiv 8 Duplikate reduziert). C1-Erfolg bei 4 Disks: 40 % (n=15, isolierte Messung innerhalb eines n=30-Laufs) — sauber im 30–50-%-Korridor. 3 Disks separat getestet und verworfen (73 % Erfolg, deutlich zu leicht; kein Schwierigkeitsgefälle innerhalb des 3-Disk-Zustandsraums nutzbar, da dessen Optimalpfad-Obergrenze bei 7 Zügen liegt). **Bewusst nicht** auf AUROC/Signalqualität kalibriert — die Konfigurationswahl beruht ausschließlich auf der Erfolgsrate; TLE/VC-Diskrimination wird als nachgelagerte, unabhängige Beobachtung in Phase 1 berichtet, nicht als Auswahlkriterium verwendet (sonst Risiko einer optimistisch verzerrten konfirmatorischen Aussage, "Garden of Forking Paths"). Verteilung optimal/legal/illegal wird aus den finalen 50 Instanzen für den Pre-Analysis-Screen dokumentiert. *Evidenz: `docs/consistency_log.md`, Einträge zur ToH-State-Diversitäts-Probe und zu den Korridor-Läufen; `data/results/gate_d_calibration/toh_corridor_scramble_n30/`.*
- [ ] **HART — Beide Manifeste final und im Freeze-Tag.** `difficulty_manifest.json` für TextWorld und ToH mit 50 Instanzen, `difficulty_tier`, `holdout` (5/45); nach diesem Punkt unveränderlich. Ein Smoke bestätigt, dass `holdout`/`difficulty_tier` im Episode-JSON ankommen (Gate B belegt den Schreibpfad bereits). **Stand 2026-07-20:** beide Manifeste existieren, sind committed und inhaltlich final (TextWorld: `r5_i1_take+cook`, Seed 20260718; ToH: 4 Disks, `random_scramble`, Seed 271828), Mock-Smoke für beide bestanden. Noch offen: der eigentliche **Freeze-Tag** (git tag) — laut diesem Dokument (Zeile 19/115) gibt es einen einzigen projektweiten Tag, gesetzt **nach Gate F**, nicht pro Gate; bewusst erst dann gesetzt, nicht vorgezogen (Entscheidung 2026-07-20, siehe `docs/consistency_log.md`). *Evidenz: Manifeste im getaggten Commit (folgt nach Gate F).*

---

## Gate E — Analyse-Rehearsal (Pipeline vor den Daten beweisen)

Begründung: Preregistrierungsdisziplin verlangt eine vor den Daten lauffähige konfirmatorische Auswertung. Jede Reparatur der Analyse nach Sichtung der Phase-1-Daten ist angreifbar.

- [x] **HART — End-to-End-Trockenlauf auf Pilot-/C-5-Daten.** Kette: Episode-JSONs → Step-Tabelle (`datasets.py`) → `grid_search_thresholds` → Policy-Artefakt → `load_policy` → `run_phase2.py`-Smoke mit `adaptive_tle` → `cluster_bootstrap` auf ΔAUROC. Jeder Übergang ohne manuelles Umformatieren. Gate B hat Teile davon (Roundtrip) bereits belegt; hier die vollständige Kette bis zum Bootstrap. **Re-verifiziert 2026-07-20** gegen aktuellen Code (der `cluster_bootstrap`-NaN-Fix vom 2026-07-17 betraf die ursprünglich berichteten Zahlen; mit Neu-Lauf bestätigt, keine offene Diskrepanz — siehe `docs/consistency_log.md`). *Evidenz: `docs/gate_e_rehearsal.md`.*
- [x] **HART — Pre-Analysis-Screen-Trockenlauf.** `preanalysis_screen.py` auf den C-5-Daten erzeugt den vollständigen Report (Signalvarianz, VC-Degeneriertheit, Clusterzahlen, Missing-VC, Klassenbalance, ICC, Bootstrap-Schiefe) ohne fehlende Eingabefelder. Keine der seit 2026-07-17 gelandeten Fixes berührt `preanalysis_screen.py`. *Evidenz: Report-JSON, `data/results/instrument_validation/phase1_20260714_105004/preanalysis_screen.{json,md}`.*
- [x] **WEICH — H3-Power-Simulation.** §5.8 sieht eine simulationsbasierte Power-Prüfung für die Interaktion vor, geseedet mit Pilot-ICC und Entropieverteilung. Nach C-5 entweder durchführen oder die Limitation aktiv wählen und in §5.9 belassen. Vor Phase 1 entscheiden, welcher Pfad gilt. **Beide Pfade genommen:** echte Monte-Carlo-Power-Simulation durchgeführt und zweimal mit besserer Evidenz neu gelaufen (2026-07-19: korrigierte TextWorld-15–40-Step-Annahme; 2026-07-20: echte ToH-Freeze-Korridor-Längen statt Pilot-Bootstrap) — siehe `docs/gate_e_h3_power_simulation.md`, Einträge auch in `docs/consistency_log.md`. Kernbefund: konfirmatorischer TextWorld/TLE-Test erreicht 80 % Power nur für Interaktionseffekte in der Größenordnung des vollen Haupteffekts; für moderate Degradation (25–50 % Abschwächung) liegt die Power bei ~42–51 % (α=.025/.05) — klar unter 80 % für den plausibelsten Fall. **Limitation aktiv in §5.9 belassen** (2026-07-20): quantifizierter Absatz in `../metacog-thesis/chapters/05_methodology.md` § 5.9 ergänzt (Commit `77bfba9`, Branch `freeze-review/instrument-validation-prose`, gepusht) — ein nicht-signifikanter H3-Befund schließt große Degradation aus, moderate bleibt plausibel/unterpowert. Damit beide im Item selbst genannten Erledigungskriterien erfüllt.

---

## Gate F — Ops und Budget

- [x] **HART — Budget-Neuschätzung aus gemessenem gebatchtem Durchsatz.** C-5 `105004` (Basiszahl, Cap 20, überholt): ~93 ep/h, N=32 → ursprünglich ~48 h GPU geschätzt. **Neu gerechnet (2026-07-20, `scripts/gate_f_budget_reestimate.py`, Token-basiertes Modell statt flacher ep/h-Zahl):** ~37–76.5 h GPU je nach Annahme für die beiden noch unvermessenen Zellen (TextWorld C1/C2 bei Cap 45, ToH-C2 bei 4 Disks) — siehe `docs/consistency_log.md`. Speicher: `action_window` Sidecars ~10–15 GB (gzip ~1–3 GB); Nicht-Holdout-`full`-Subset (Inst. 1–3) low tens GB. *Evidenz: Consistency-Log @ `9f8dafd` + Sidecar-Policy-Eintrag; Re-Schätzung @ `84267d7`.* **Kosten vs. Restguthaben: erledigt/akzeptiert (2026-07-20).** Guthaben aufgestockt (aktuell ~36 EUR), Zeit ist jetzt der limitierende Faktor, nicht Geld — bewusste Nutzerentscheidung, kein weiterer Top-up-Blocker vor Phase 1.
- [x] **HART — Resume-Korrektheit unter Nebenläufigkeit (angepasst wegen Parallelisierung).** Nicht mehr nur „max. eine Episode verloren": Bei parallelisiertem Backend sind zum Crash-Zeitpunkt mehrere Episoden in flight. Test: laufenden gebatchten Smoke hart abbrechen, mit `--resume` fortsetzen; kein halb geschriebenes Episode-JSON, kein Doppel-Eintrag, kein übersprungenes Work-Item über `list_completed_episodes`. Schwere Crash-Resilienz ist per Projektentscheidung deprioritisiert, aber die Resume-Korrektheit im nebenläufigen Fall bleibt HART, weil jetzt mehr gleichzeitig offen ist. **Durchgeführt (2026-07-20), lokal mit Mock-Backend** (Korrektheit hängt rein an Bookkeeping/Threading, nicht am Backend — kein Pod nötig): echtes `SIGKILL` gegen laufenden `run_phase1.py`-Batch fand einen realen Bug — `log_episode()` schrieb `json.dump()` direkt auf den finalen `ep_*.json`-Pfad ohne Temp-Datei/Rename; ein Kill mitten im Schreiben hinterließ eine abgeschnittene Datei, die `list_completed_episodes()` (reiner Existenz-Glob, keine Inhaltsprüfung) dauerhaft als "fertig" zählte und nie neu lief. Reproduziert (2 unabhängige Batch-Zyklen, 7/10 im gezielten Write-Race-Probe), gefixt via Temp-Datei + `os.replace()` (atomar). Nach Fix: 0/8 Batch-Zyklen, 0/10 Probe-Trials zeigen die Race noch; Regressionstest ergänzt. *Evidenz: `scripts/gate_f_resume_smoke.py`, PR #23, `docs/consistency_log.md`.*
- [ ] **HART — Run-Hygiene auf dem Pod.** Modell vorab auf der ephemeren Container-Disk gecacht (bewusst **nicht** auf dem persistenten Network Volume — RunPods Template-Default zeigt `HF_HOME` fälschlich auf `/workspace/.cache`, `scripts/pod_runtime_env.sh` korrigiert das; Network Volume bleibt klein, nur Code + Results, Re-Download nach Container-Neustart ist einkalkuliert); Langfuse-Keys gesetzt oder Tracing bewusst aus (dokumentiert); Repo-Checkout selbst liegt unter `/workspace` (das ist der eigentliche Persistenz-Mechanismus, nicht die `RESULTS_DIR`-Env-Var, die von keinem Script gelesen wird); History-Guard aktiv; **Logprob-Sidecars:** `logprob_sidecar_mode: action_window`, **Reasoning-full** nur für Nicht-Holdout-Instanzen in `logprob_sidecar_full_instances` (z. B. 1–3, nicht Holdout mod-10); parallelisiertes Backend mit den in C-1 als batch-invariant verifizierten Einstellungen (N=32). **Preflight-Script:** `scripts/gate_f_run_hygiene_preflight.py` prüft alle fünf Punkte automatisiert, ohne GPU/Inferenz. *Evidenz: `run_metadata` des ersten Phase-1-Blocks + Preflight-Report — ausstehend.*
- [ ] **HART — C1/C2-Qualitätskontrolle unter den aktuellen, kalibrierten Korridoren.** Gate C-2 (Format-Compliance) lief auf einem kurzen Smoke, vor der Gate-D-Kalibrierung und vor dem `max_steps`-Fix (`fix/toh-per-instance-max-steps`) — die jetzt viel längeren Episoden (TW Cap 45, ToH bis 45 pro Instanz) sind seitdem nie mit echtem C1/C2-Reasoning durchlaufen worden. Zu prüfen an einer kleinen echten Stichprobe (n≈10–20/Domäne/Stufe): Think-Block sauber geschlossen, Aktion nach `</think>` parsebar, keine Reasoning-Leakage in den Action-Slot, kein systematisches Anschlagen an `cot_max_tokens=8192` (würde auf zu knappes Budget hindeuten), C2 effektives N plausibel (Pilot-Referenz ~2.9/3). **Bewusst gebündelt mit der Run-Hygiene-Pod-Session** (User-Entscheidung 2026-07-20) — kein separater Pod-Trip nötig. *Evidenz: ausstehend, Probe-Report von der Run-Hygiene-Session.*
~~**WEICH — Block-Aufteilung.**~~ **Gestrichen (2026-07-20, Nutzerentscheidung).** Beide ursprünglichen Gründe (Resume-Grenzen, Fehlerisolation) sind seit dem Gate-F-Resume-Fix bereits auf Episoden-Ebene gelöst (atomares Schreiben, korrektes `--resume`), unabhängig davon, ob Domänen getrennt oder gemischt laufen. Eine echte Domain-Trennung hätte zusätzliche Komplexität gebraucht (neuer `--domains`-CLI-Flag, Risiko doppelter N=32-Überlappung bei parallelen Blöcken) ohne echten Mehrwert gegenüber dem einfachen Beobachten/ggf. Killen+Resumen des laufenden Single-Runs. *Begründung: `docs/consistency_log.md`.*

---

## Akzeptierte, nicht-blockierende Restrisiken (benannt, nicht gelöst)

1. **LM-Studio-Paritätskriterium 3 (Cross-Backend-Entropiegleichheit) = not_applicable** wegen Modellasymmetrie (4B-MLX lokal vs. 8B-vLLM Pod); tragend sind Kriterien 1+2+Batch-Invarianz auf vLLM.
2. **Lost-in-the-middle** unter Full-History (§5.9, Liu et al., 2023): bewusster Trade-off der H3-Konfundierungskontrolle. Die 5090 mit 32 GB erlaubt großzügigeres `max_model_len`, was die Full-History-Anforderung technisch entspannt.
3. **VC-Kostenasymmetrie** in Phase 1 (§5.3/§5.9): strukturell, benannt.
4. **H4-Observability-Konfound** (§5.5.2/§5.9): benannt, Interpretation gebunden.
5. **5-Instanzen-Holdout-Varianz** (§5.9): durch Pooling und Sensitivitätsbericht mitigiert.
6. **Solution-Space-Kompression** am Episodenende als H3-Teilkonfound: durch Step-Level-Labels partiell kontrolliert.
7. **Nichtdeterminismus unter Parallelisierung:** durch das Batch-Invarianz-Kriterium in C-1 auf ein preregistriertes Toleranzniveau begrenzt; Restrauschen unterhalb der Toleranz ist akzeptiert und dokumentiert.

---

## Go/No-Go

| Gate | Inhalt | Status |
|------|--------|--------|
| A | Preregistrierungs-Freeze (Prosa) | **abgeschlossen** (`c0d4067`) |
| B | Code-Readiness | **abgeschlossen** (`6ca2857`) |
| C | Instrument-Validierung (5090, parallel vLLM): C-0…C-6 | **abgeschlossen** (`9f8dafd`) |
| D | Schwierigkeitskalibrierung + Manifeste | **teilweise** (beide Korridor-HART-Punkte abgehakt, Manifeste inhaltlich final; einziger Rest ist der projektweite Freeze-Tag nach Gate F) |
| E | Analyse-Rehearsal | **abgeschlossen** (beide HART-Punkte + WEICH-Punkt erledigt; HART-Punkte am `2026-07-20` gegen aktuellen Code re-verifiziert, WEICH-Punkt H3-Power-Simulation durchgeführt und in §5.9 dokumentiert) |
| F | Ops + Budget (5090, Batching) | **teilweise** (Budget/Speicher + Top-up abgeschlossen; Resume-Korrektheit abgeschlossen (PR #23); Run-Hygiene + C1/C2-QC offen, beide gebündelt in der nächsten Pod-Session; Block-Aufteilung gestrichen) |

**GO** = alle HART-Punkte abgehakt, jedes Evidenz-Artefakt archiviert, WEICH-Punkte erledigt oder mit Datum und Begründung als akzeptiert eingetragen. Danach: Freeze-Tag setzen, ersten Phase-1-Block starten, dieses Dokument mit dem Tag-Hash abschließen. Änderungen danach nur als datiertes Amendment.

---

## Was sich gegenüber rev. 1 geändert hat (Änderungsnachweis)

- GPU von RTX 3090 (Header) bzw. 4090 (Gate-B-Log) auf **RTX 5090** vereinheitlicht; die frühere GPU-Diskrepanz als offener Gate-F-Punkt ist damit **erledigt und gestrichen**.
- Gate C um ein drittes Paritätskriterium **Batch-Invarianz** (C-1.3) erweitert, weil das Backend jetzt parallelisiert (Continuous Batching) läuft.
- Gate-F-Resume-Punkt von „max. eine Episode verloren" auf **Resume-Korrektheit unter Nebenläufigkeit** umformuliert.
- Durchsatz- und Budgetgrundlage von sequenziell/3090 auf **gebatcht/5090** umgestellt.
- Gate A und B als abgeschlossen markiert mit Commit-Nachweis.
- Frühbefund zum degenerierten TLE/VC aus der realen Smoke-Fixture als Vorbemerkung aufgenommen; C-4 und C-5 als Kipp-Punkte gekennzeichnet.
