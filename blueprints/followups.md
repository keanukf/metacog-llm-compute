# Follow-ups / später (nicht Smoketest-blocking)

Diese Punkte sind bewusst **nicht** Teil des RunPod-Smoketests (Instrumentierung + E2E-Pipeline),
sondern sollten vor Phase 1/2 (größere Runs, längere Episoden, Kosten) separat behandelt werden.

## 1) Prompt-Context Safety-Cap (History wächst aktuell unbounded)

Aktueller Zustand:
- In `configs/pilot.yaml` ist für TextWorld `history_max_obs_chars: 0` und `history_current_obs_max_chars: 0` gesetzt (Truncation aus).
- Das ist für Debugging lokal sinnvoll, aber bei Phase 1/2 (mehr Steps, C2 Best-of-N) kann die Prompt-Länge schnell explodieren.

Empfehlung:
- Ein **Safety-Limit** implementieren, das unabhängig von Experiment-Parametern greift (z.B. harte Obergrenze in Tokens/Chars), plus Logging, wenn die Kappe greift.
- Optional: „structured“ Kompression (z.B. nur `Valid commands`, Inventory, zuletzt relevante Objekte), aber erst nach Baseline-Resultaten.

## 2) TextWorld `correctness` (legal/illegal) nutzt falsche Admissible-Liste

Symptom:
- Aktionen können erfolgreich sein (Raumwechsel), werden aber als `illegal` markiert.

Ursache (wahrscheinlich):
- `TextWorldEnv.step()` prüft `admissible_commands` aus dem **post-step** `info` (neuer Zustand), nicht aus dem Zustand *vor* dem Action-Apply.
- Dadurch ist das eingereichte Kommando oft nicht mehr in der neuen Admissible-Liste.

Empfehlung:
- Admissible für „legal check“ aus dem pre-step State beziehen (falls TextWorld das zulässt), oder ein alternatives Kriterium verwenden:
  - Parser-Fehler/Feedback-String prüfen (falls verfügbar), oder
  - `info["admissible_commands"]` im pre-step separat abfragen/cachen (Reset/Step return values).
- Danach Tests hinzufügen: Transition-Action muss als `legal` gelten, wenn Observation sich erwartbar ändert.

