# Gate D — Parser-Audit (TextWorld, r3_i1_take-only)

**Datum:** 2026-07-15  
**Quelle:** Replay-Inspektion `textworld_abort_action_inspection/abort_action_inspection.json`  
(8 Episoden, gleiche Seeds wie Sweep v1; letzte 4 Steps pro Episode; `--real`, Cap 25)

**Operationalisierung der roten Linie:**  
Eine Änderung ist erlaubt, wenn sie nur bestimmt, ob eine **bereits korrekt gemeinte** legale Aktion durch den Parser kommt — nicht, welche Aktion das Modell wählt. Keine admissible commands im Prompt/Observation. Identisch über alle Zellen und Domänen (TW-only für Synonym-Map; ToH unberührt).

---

## Kategorie A — Synonym legaler Aktionen (behebbar)

| Instanz | Step | Modell-Output | Befund | Fix |
|--------:|-----:|-----------------|--------|-----|
| 7 | 21–24 | `look inventory` (×4) | Korrekte Absicht (Inventar), falsche Oberflächenform; Parser kennt `inventory`, nicht `look inventory`. Restdistanz 2→2. | **Ja:** `look inventory` → `inventory`, nur wenn `inventory` pre-step admissible |

**Keine weiteren Synonym-Fälle in der Stichprobe.**  
(`prepare meal` / `eat meal` in Instanz 2: bereits legal/optimal wenn admissible — kein Syntax-Artefakt.)

---

## Kategorie B — Planungsfehler / falsche Aktion (nicht anfassen)

| Instanz | Step | Modell-Output | Befund |
|--------:|-----:|-----------------|--------|
| 1 | 21–24 | `fry yellow onion with stove` (×4) | Halluziniertes Verb + Setup ohne Koch-Station (take-only). Kein Synonym für eine legale Aktion — Modell weiß nicht, was zu tun ist. |
| 2 | 15 | `take sugar` | Illegale Objekt-Aktion trotz nahem Finish — falsche Wahl, kein Vokabular-Mismatch. |
| 0 | 23 | `go west` | Richtung nicht admissible — Navigationsfehler. |
| 3 | 23 | `go north` | Richtung nicht admissible. |
| 5 | 21–22 | `go east`, `go west` | Richtungen nicht admissible. |
| 6 | 21–24 | `go north` (×4) | Wiederholtes illegales Kommando — Kreisen / falscher Plan. |

---

## Kategorie C — Navigation ohne Abschluss (nicht anfassen)

| Instanz | Muster | Befund |
|--------:|--------|--------|
| 0, 3, 4, 5 | Legal/optimal `go *`, teils dist↓, kein Win | Echtes Planungs-/Suchverhalten; Decke trifft teils, löst Finish nicht. |
| 5 | Letzte Steps optimal 4→3→2 | Fortschritt, aber kein Finish — kein reines Syntax-Problem. |

---

## Explizit nicht erlaubt (DV-Schutz)

- `fry`/`cook`-Halluzinationen in take-only unterdrücken oder umlenken  
- Schlusssequenz (`prepare meal`, `eat meal`) soufflieren  
- `include_admissible_commands: true` oder state-spezifische Kommandolisten im Prompt  
- Zell- oder domain-spezifische Prompt-Unterschiede  

---

## Geplanter Fix (nur Kategorie A)

1. Statische Synonym-Tabelle in `textworld_env.py` (TW-only, confidence-neutral).  
2. Rewrite **nur** wenn kanonische Form in pre-step `admissible_commands`.  
3. `action_raw` = Modell-Output unverändert; ausgeführt wird kanonische Form.  
4. Kontrollmessung: 8× `r3_i1_take-only` nach Fix; Erfolgsrate vs. Baseline (0–1/8 im Replay).  

**Erwartung:** Modest lift (Instanz 7 class), nicht Sprung auf Korridor — Rest bleibt Kategorie B/C.

---

## Fix implementiert (2026-07-15)

**Code:** `textworld_env._resolve_synonym_for_admissible` — nur Kategorie A.  
**Kontrollmessung** (`textworld_abort_action_inspection_post_synonym/`):

| | Pre-Fix | Post-Fix |
|---|--------:|---------:|
| Erfolg r3_i1_take-only (8 ep) | 1/8* | **0/8** |

\*Pre-Inspect-Replay; Trajektorien nicht bit-identisch (Inference-Varianz).

**Instanz 7:** `look inventory` → ausgeführt als `inventory`, Label **legal** (vorher illegal). Quest-Distanz bleibt 2→2 — Synonym hebt Syntax-Label an, ersetzt kein Finish.

**Fazit:** Fix tut das Richtige für Kategorie A; Erfolgsrate bleibt am Boden → echter Planungs-/Finish-Befund (B/C) dominiert. Schwierigkeitskalibrierung erst auf dieser Basis.
