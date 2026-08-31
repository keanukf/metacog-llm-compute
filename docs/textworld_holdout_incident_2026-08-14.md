# Chronik: TextWorld-Holdout-Inkonsistenz (entdeckt und behoben 2026-08-14)

Eigenständige, lesbare Zusammenfassung eines Vorfalls, der sich über mehrere Wochen und mehrere
Analyseschritte erstreckt. Für die tagesgenaue, technische Beweisführung siehe
`docs/consistency_log.md` (Einträge 2026-08-14, zwei Stück, neuester zuerst). Dieses Dokument ist
die Kurzfassung zum schnellen Wiedereinstieg, nicht der Ersatz für den Log.

## TL;DR

Die TextWorld-Holdout-Instanzen (die 5 von 50 Instanzen, die für das Fitten der Phase-2-Allocator-
Schwellenwerte reserviert sein sollten) waren in der tatsächlich verwendeten Datenquelle falsch
gesetzt — nicht durch eine Design-Entscheidung, sondern durch einen Daten-Labeling-Bug in einer
Zwischensammlung vom Juli. Das Manifest selbst war immer korrekt. Der Fehler betrifft ausschließlich
TextWorld (Tower of Hanoi ist sauber) und wirkt sich in zwei getrennten Formen aus: eine
**statistische Zirkularität** in der H2-Konfirmatorik (behoben, sauber, nachvollziehbar) und eine
**Protokollabweichung** beim tatsächlich deployten Phase-2-Allocator (nicht mehr behebbar ohne
Neusammlung, die aus Kostengründen nicht in Frage kommt — vollständig transparent offengelegt statt
verschwiegen). Beide Aspekte sind jetzt in Code, Daten-Reruns und Thesis-Prosa sauber
auseinandergehalten.

## Zeitlicher Ablauf

**2026-07-18 — Gate D, Manifest eingefroren.** `data/tasks/textworld/difficulty_manifest.json`
wird erzeugt und committet (Commit `e34853b`), `holdout_policy: mod-10` → Holdout-Instanzen
`{0, 10, 20, 30, 40}`. Dieser Wert ändert sich danach nie wieder — das Manifest ist zu jedem
späteren Zeitpunkt korrekt.

**2026-07-22 — erste Phase-1-Sammlung.** `data/results/phase1/phase1_20260722_091125/` läuft
vollständig, mit für TextWorld korrekten Holdout-Labels (übereinstimmend mit dem Manifest). Aber:
45 von 50 TextWorld-Spieldateien fehlen zur Laufzeit auf dem Collection-Pod, `TextWorldEnv` fällt
still auf einen unlösbaren Stub zurück statt einen Fehler zu werfen (5,3% TextWorld-Erfolgsrate in
diesem Ordner vs. 53,7% in der folgenden Neusammlung — der empirische Fingerabdruck des Stub-Bugs).
Der Fix dafür (Commit `b47e35d`, "fail loudly when TextWorld game files are missing") landet, aber
der bereits gesammelte TextWorld-Teil dieses Laufs ist damit für TextWorld unbrauchbar. ToH aus
diesem Lauf bleibt gültig und ist bis heute die kanonische ToH-Quelle.

**2026-07-24/25 — Neusammlung für TextWorld.** `data/results/phase1/textworld_regen_20260724/`
läuft nach dem Stub-Fix erfolgreich durch und wird zur kanonischen TextWorld-Quelle
(`src/analysis/phase1_canonical.py`). **Hier entsteht der eigentliche Bug**: die Sammlung stempelt
ihr `holdout`-Feld pro Episode nicht gegen das offizielle Manifest, sondern gegen eine andere,
nie committete Konfigurationsdatei (`configs/dev/textworld_regen_real.yaml`, laut
`run_metadata.json` dieser Sammlung — die Datei selbst existiert nicht mehr im Git-Verlauf, der
genaue Codepfad, der die falsche Split-Logik erzeugt hat, ist nicht mehr rekonstruierbar). Ergebnis,
empirisch verifiziert: `holdout=True` für Instanzen `{0,1,2,3,4}` (die ersten fünf) statt für die
Manifest-Instanzen `{0,10,20,30,40}`. Zu diesem Zeitpunkt fällt das niemandem auf — die Sammlung
läuft technisch fehlerfrei durch, die Zahlen sehen plausibel aus.

**Anfang/Mitte August 2026 — Threshold-Artefakt gebaut, Phase 2 vorbereitet.**
`scripts/phase2_prep/build_threshold_artifact.py` fittet die TextWorld-Allocator-Schwellenwerte
gegen die (falsch gelabelten) Holdout-Steps aus der Regen-Sammlung. Das Skript vertraut dem
eingebetteten `holdout`-Feld der Episoden, statt es gegen das aktuelle Manifest neu abzuleiten — zu
diesem Zeitpunkt ein plausibles, nicht offensichtlich falsches Design (das Feld *sollte* ja das
Manifest widerspiegeln). Die deployten TextWorld-Schwellenwerte lauten in der Folge θ₁=0,70/θ₂=0,80
(TLE) und θ₁=0,80/θ₂=0,90 (VC) — beide, wie sich später herausstellt, deutlich enger/konservativer
als das, was die korrekten Holdout-Instanzen ergeben hätten.

**2026-08-05 bis 2026-08-10 — Phase-2-Sammlung.** Läuft mit den (falsch gefitteten) Schwellenwerten.
TextWorld wird für alle sechs Strategien vollständig gesammelt (750/750 pro Zelle). Parallel dazu
läuft ein komplett unabhängiges Problem: ein Speicherleck/OOM-Muster bricht die
`always_c2`/Tower-of-Hanoi-Sammlung bei 12 von 50 Instanzen ab (eigene Chronik in
`docs/consistency_log.md`, Eintrag 2026-08-10) — dieser zweite Vorfall betrifft eine andere Domäne
und ein anderes Subsystem, hat mit dem Holdout-Bug nichts zu tun, koexistiert aber zeitlich mit ihm.

**2026-08-13/14 — Entdeckung.** Beim Schreiben eines explorativen Always-C0/Always-C1-
Vergleichs (der die Phase-1-Fixed-Stage-Daten mit den Phase-2-Daten kombiniert) fällt eine
Diskrepanz auf: die TextWorld-Stichprobe hat $N=41$ statt der erwarteten 45
($=50-5$ präregistrierte Holdout-Instanzen). Diese Anomalie — nicht eine Vermutung, sondern ein
konkretes, falsifizierbares Zahlen-Mismatch — löst die Untersuchung aus.

**2026-08-14 — Root-Cause-Analyse.** Systematische Prüfung mit Primärbelegen (nicht Spekulation):
`git log --follow` auf das Manifest (ein einziger Commit, immer korrekt), direkte JSON-Inspektion
aller 750 TextWorld-Episoden in allen drei Rohdatenquellen (ursprüngliche Sammlung, Regen-Sammlung,
Phase-2-Sammlung), Code-Inspektion jedes Skripts, das das `holdout`-Feld konsumiert. Ergebnis: die
Regen-Sammlung ist die einzige fehlerhafte Quelle; sie ist aber die kanonische TextWorld-Quelle, was
den Fehler in die tatsächlich deployte Allocator-Konfiguration hineinträgt. Genaue Betroffenheits-
Analyse: 4 Instanzen (`1,2,3,4`) wurden unter dem falschen Schema zum Fitten verwendet und liegen
unter dem korrekten Schema *nicht* im Holdout — sie sind daher sowohl in der Fitting- als auch in
der Phase-2-Konfirmatorik-Stichprobe enthalten (die eigentliche Zirkularität). Instanz `0` und die
Instanzen `10,20,30,40` sind auf ihre jeweils eigene Art unproblematisch (siehe
`docs/consistency_log.md` für die vollständige Herleitung).

**2026-08-14 — Fix, umgesetzt und deployt.** Nutzer-Freigabe: "Ja, mach das bitte so." Umsetzung:
- `apply_textworld_holdout_correction()` (`src/analysis/phase1_canonical.py`), zwei benannte
  Instanzmengen (`TEXTWORLD_TRUE_HOLDOUT_INSTANCES` für Fitting-Kontexte,
  `TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES` für Evaluations-Kontexte der bereits deployten
  Phase-2-Policy), verdrahtet in alle vier betroffenen Analyseskripte.
- Threshold-Artefakt, H1b, H2 und der C0/C1-Referenzvergleich neu gerechnet (482 Tests grün).
- Kapitel 6 (§6.2), Kapitel 7 (Epigraph, Tabellen 7.1/7.2, neuer §7.3) und Kapitel 8 (§8.1, §8.4,
  §8.5) der Thesis entsprechend aktualisiert, inklusive expliziter Offenlegung der nicht mehr
  behebbaren Protokollabweichung.

**2026-08-14 — Sensitivitätsanalyse nachgeliefert.** Kapitel 5 §5.9 hatte bereits vor diesem
Vorfall eine "reported sensitivity analysis around the selected thresholds" angekündigt (Begründung
dort: fünf Holdout-Instanzen sind eine kleine, varianzanfällige Stichprobe), aber nie geliefert. Der
Holdout-Bug liefert dafür einen realen statt hypothetischen Vergleichsfall. Neues Skript
`scripts/phase2_prep/threshold_sensitivity_analysis.py` (+ Test) rekonstruiert die deployte und die
korrekte Konfiguration beide aus Code heraus (keine Zahl von Hand übernommen) und vergleicht sie auf
zwei Arten: wo das deployte Schwellenwertpaar auf dem korrekt ausgewerteten Pareto-Front landet, und
wie unterschiedlich beide Policies die tatsächlich beobachteten Phase-2-Signalwerte geroutet hätten.
Kernbefund: die deployte VC-Policy routete 97,7% aller TextWorld-Schritte auf C0 (Direktinferenz) —
praktisch Always-C0, nicht die vorgesehene adaptive Policy. Details und volle Zahlen:
`docs/consistency_log.md`, Eintrag "Threshold-Sensitivitätsanalyse".

## Was sich am Design/an der Analyse geändert hat, und warum

| Änderung | Warum | Wo dokumentiert |
|---|---|---|
| TextWorld-H2-Konfirmatorik: $N=45 \to 41$ | Entfernt die Fit/Eval-Überlappung für Instanzen 1-4 | `07_results_adaptive_allocation.md` (Epigraph, Tabelle 7.1) |
| H1b TextWorld (ΔBrier) neu berechnet | Kalibrator war auf falschen 5 statt präregistrierten 5 Instanzen gefittet (nie zirkulär, aber falsche Instanzen) | `06_results_signal_quality.md` §6.2 |
| Threshold-Artefakt neu gebaut | Für die Akten / für die Sensitivitätsanalyse — ändert **nicht** die bereits deployten Phase-2-Daten | `data/results/phase1/threshold_artifact.json`, `consistency_log.md` |
| Neuer §7.3 (Always-C0/C1-Spektrum) | War bereits vom Nutzer freigegeben, aber vor der Holdout-Entdeckung nie in Prosa geschrieben; jetzt mit korrigierten Zahlen nachgeholt | `07_results_adaptive_allocation.md` |
| Neue Threshold-Sensitivitätsanalyse | In §5.9 bereits angekündigt, nie geliefert; der Holdout-Bug lieferte den konkreten Anlass | `scripts/phase2_prep/threshold_sensitivity_analysis.py`, `consistency_log.md` |

## Was bleibt unkorrigierbar (offen gelegt, nicht versteckt)

Die bereits gesammelten `adaptive_tle`/`adaptive_vc`-Episoden für TextWorld liefen mit den falsch
gefitteten Schwellenwerten (θ=0,70/0,80 bzw. 0,80/0,90 statt korrekt 0,10/0,90). Das lässt sich ohne
Neusammlung nicht rückwirkend reparieren; eine Neusammlung wurde aus denselben Kosten-/Zeitgründen
nicht angestoßen, die auch den ToH-Always-C2-Abbruch vom 2026-08-10 begründet haben. Das
TextWorld-H2-Ergebnis ist damit konfirmatorisch für **die tatsächlich deployte Policy** bei
$N=41$, nicht für die in Kapitel 5 exakt spezifizierte Policy bei $N=50$ — eine Protokollabweichung
mit bekannter, quantifizierter Ursache, kein unbekanntes Risiko.

## Betroffenheit im Überblick

| Hypothese/Analyse | Betroffen? | Status |
|---|---|---|
| H1a (Diskrimination) | Nein | kein Holdout-Gebrauch |
| H1b (Kalibrierung) | Ja, aber nie zirkulär | neu berechnet, Befund unverändert |
| H2 TextWorld (konfirmatorisch) | Ja, Zirkularität | korrigiert, $N=41$, Befund unverändert |
| H2 Tower of Hanoi (explorativ) | Nein | Manifest und Daten stimmten dort immer überein |
| H3 (temporale Degradation) | Nein | kein Holdout-Gebrauch |
| H4 (Domänenkontrast) | Nein | kein Holdout-Gebrauch |
| C0/C1-Explorativvergleich | Ja | korrigiert (war der ursprüngliche Auslöser der Entdeckung) |
