# Infrastrukturplan, Budget & Feasibility-Pilotstudie (v2)

## Master Thesis: Metacognitive Effort Allocation in Sequential Language Model Agents
### Keanu Forthmann — Infrastruktur- und Durchführungsplan

---

## I. Hardware-Setup

### Vorhandenes Equipment

| Ressource | Spezifikation | Rolle im Projekt |
|-----------|--------------|------------------|
| MacBook Pro M1 | 8/16GB Unified Memory | Entwicklung, Debugging, lokale Tests, Analyse |
| Home Server (Proxmox) | Viel RAM, keine dedizierte GPU | Datenbackup, Logging-Archiv |

### Was das Experiment braucht

Qwen2.5-3B-Instruct benötigt ~6 GB VRAM (FP16). Inferenzgeschwindigkeit mit vLLM:
- RTX 3090 (Cloud): **~100–150 tok/s**
- M1 lokal (llama.cpp/MLX): **~20–40 tok/s**

Lokal wäre das Experiment ~5× langsamer — Cloud-GPU für die Hauptexperimente, M1 für alles andere.

---

## II. Cloud-Plattform: RunPod Community Cloud

### Account-Status

- [x] RunPod-Account erstellt
- [x] $10 Guthaben aufgeladen (reicht für Pilot + erste Core-Runs)

### Gewählte Konfiguration

| Parameter | Wert |
|-----------|------|
| **GPU** | RTX 3090 (24 GB VRAM) |
| **Pricing** | $0.22/hr (Spot) — sekundengenaue Abrechnung |
| **Template** | RunPod PyTorch 2.x |
| **Volume** | 30 GB Network Volume (Modell + Ergebnisse persistent) |

**Warum RTX 3090 statt 4090:** Für Inferenz eines 3B-Modells ist die 3090 mehr als ausreichend. Die 4090 bringt ~50% mehr Speed, kostet aber ~75% mehr ($0.39/hr). Preis-Leistung gewinnt die 3090.

**Warum Spot statt On-Demand:** Spot ($0.22) statt On-Demand ($0.36) spart 39%. Das Risiko einer Unterbrechung wird durch Checkpointing pro Episode eliminiert — bei Abbruch geht maximal 1 Episode verloren.

---

## III. Compute-Budget: Detailberechnung (v2 Design)

### A. Inferenz-Parameter

| Parameter | Wert |
|-----------|------|
| Modell | Qwen2.5-3B-Instruct (FP16) |
| Inferenz-Backend | vLLM |
| Geschwindigkeit (konservativ) | 120 tok/s |
| Tokens pro LM-Call (Ø) | 200 (Input ~120, Output ~80) |
| Temperatur | 0.3 |
| Runs pro Instanz/Bedingung | 5 |
| Steps pro Episode (Ø) | 10 |

### B. Phase 1 — Signal Calibration

Jede Instanz wird in 3 separaten Durchläufen (je 1 pro Compute-Stufe) absolviert. Im C0-Durchlauf werden TLE (via Logprobs, kostenlos) und VC (1 Extra-Call) erfasst.

| Compute-Stufe | LM-Calls/Step | Begründung |
|---------------|---------------|-----------|
| C0 + Signale | 2 | 1 Action (+ Logprobs → TLE) + 1 VC-Prompt |
| C1 (CoT+Verify) | 2 | 1 CoT-Generation + 1 Self-Verification |
| C2 (Self-Consistency, N=3) | 3 | 3 parallele Generierungen + Majority Vote |

| Komponente | Berechnung | Ergebnis |
|-----------|-----------|---------|
| Instanzen | 50/Domäne × 2 Domänen | 100 |
| Episoden | 100 × 3 Stufen × 5 Runs | **1.500** |
| Calls total | (500×2 + 500×2 + 500×3) × 10 Steps | **35.000** |
| Tokens | 35.000 × 200 | 7M |
| **GPU-Zeit** | 7M ÷ 120 tok/s | **~16 Stunden** |

### C. Phase 2 — Adaptive Allocation

6 Strategien werden verglichen. Calls/Step variieren je nach Strategie:

| Strategie | Calls/Step (Ø) | Episoden (100 Inst. × 5 Runs) |
|-----------|---------------|-------------------------------|
| Adaptive-TLE | ~2.5 (Signal + variable Stufe) | 500 |
| Adaptive-VC | ~3.0 (Signal + VC-Prompt + variable Stufe) | 500 |
| Always-C0 | 1.0 | 500 |
| Always-C2 | 3.0 | 500 |
| Random | 2.0 (Ø über Stufen) | 500 |
| EAGer-Style | 2.0 (Ø, feste Stufe pro Episode) | 500 |

| Komponente | Berechnung | Ergebnis |
|-----------|-----------|---------|
| Episoden total | 100 × 6 × 5 | **3.000** |
| Calls total (gewichtet) | 3.000 × 10 × Ø2.25 | **~67.500** |
| Tokens | 67.500 × 200 | 13.5M |
| **GPU-Zeit** | 13.5M ÷ 120 tok/s | **~31 Stunden** |

### D. Core-Gesamtbedarf

| Posten | GPU-Stunden |
|--------|-------------|
| Phase 1 — Calibration | 16 |
| Phase 2 — Adaptive Allocation | 31 |
| Overhead (TextWorld-Execution, I/O, Logging) | +15% |
| Debugging & Reruns | +10% |
| **Core Total** | **~59 GPU-Stunden** |

**Reduktion gegenüber v1:** 108 → 59 Stunden (−45%). Hauptgrund: Tree Search (C3, ~14 Calls/Step) ersetzt durch Self-Consistency Sampling (C2, 3 Calls/Step).

### E. Extensions

| Erweiterung | Zusätzliche GPU-Stunden | Kosten |
|------------|------------------------|--------|
| Semantic Consistency (5× Sampling/Step) | +20 | $4.40 |
| Zweites Modell (Phi-3.5-mini) | +55 | $12.10 |
| Dritte Domäne (Logical Reasoning) | +30 | $6.60 |
| **Extension Total** | **~105** | **$23.10** |

### F. Gesamtkosten nach Szenario

| Szenario | GPU-Stunden | Kosten ($0.22/hr) | in € (ca.) |
|----------|------------|-------------------|------------|
| **Pilot** | 2 | $0.44 | **€0.40** |
| **Core Only** | 59 | $12.98 | **€12** |
| **Core + SC-Signal** | 79 | $17.38 | **€16** |
| **Core + alle Extensions** | 164 | $36.08 | **€33** |
| **Alles + 30% Buffer** | 213 | $46.86 | **€43** |

### G. Budget-Allokation

| Posten | Kosten (€) |
|--------|-----------|
| Pilot ($10 bereits geladen) | ~€0.40 |
| Core-Experiment | ~€12 |
| Extensions (nach Bedarf) | €4–21 |
| Storage (RunPod Volume) | ~€2 |
| Buffer für Reruns | ~€8 |
| **Gesamtausgaben (realistisch)** | **€25–45** |
| **Verbleibend von $10 Startguthaben** | ~$9.50 nach Pilot |

**$10 reicht für Pilot + Core.** Für Extensions einfach bei Bedarf $20–30 nachladen. Gesamtkosten bleiben weit unter €100.

---

## IV. 6-Wochen-Durchführungsplan

### Überblick

```
Woche 1: Pilot + Code-Entwicklung (lokal + 1 Cloud-Test)
Woche 2: Task-Design + Agent-Implementation + lokale Tests
Woche 3: Phase 1 — Calibration Experiment (Cloud, Nacht-Runs)
Woche 4: Phase 2 — Adaptive Allocation Experiment (Cloud, Nacht-Runs)
Woche 5: Extensions + Analyse + Visualisierung
Woche 6: Schreiben (Methodik + Ergebnisse + Diskussion)
```

### Detail-Aufschlüsselung

**Woche 1 (17.–23. Feb): Pilot & Infrastruktur-Validierung**

| Tag | Aufgabe | Wo |
|-----|---------|-----|
| Mo–Di | Pilot-Studie (siehe Abschnitt V) | RunPod Cloud |
| Mi | Ergebnisse analysieren, Compute-Schätzung validieren | Lokal |
| Do | Codebase-Skelett: Agent-Loop, Logging, Config-System | Lokal (M1) |
| Fr | TextWorld-Installation + Task-Generator-Prototyp | Lokal (M1) |
| Sa–So | Delayed-Cue-Task-Generator implementieren | Lokal (M1) |

**Woche 2 (24. Feb – 2. März): Implementation**

| Tag | Aufgabe | Wo |
|-----|---------|-----|
| Mo–Di | Signal-Extraktion (TLE via Logprobs + VC-Parsing) implementieren | Lokal (M1) |
| Mi–Do | Compute-Stufen (C0, C1-CoT+Verify, C2-Self-Consistency (N=3)) implementieren | Lokal (M1) |
| Fr | Adaptiver Allokator + EAGer-Style-Baseline implementieren | Lokal (M1) |
| Sa–So | 50 Task-Instanzen pro Domäne generieren + validieren | Lokal (M1) |

**Woche 3 (3.–9. März): Phase 1 — Calibration (Cloud-Runs)**

| Slot | Aufgabe | GPU-Zeit |
|------|---------|----------|
| Mo Nacht | Pod starten, Environment validieren, Dry-Run (5 Instanzen) | 1h |
| Di Nacht | Phase 1: TextWorld (50 Inst. × 3 Stufen × 5 Runs) | ~8h |
| Mi Nacht | Phase 1: Delayed-Cue (50 Inst. × 3 Stufen × 5 Runs) | ~8h |
| Do–Fr | Daten herunterladen, erste Calibration-Analyse (ECE, Brier) | Lokal |
| Sa–So | Schwellenwert-Optimierung für Allokator auf Validierungsdaten | Lokal |

*Strategie: Experiment-Scripts starten abends (~20 Uhr), laufen über Nacht, Pod stoppt automatisch. Pod tagsüber gestoppt = keine Kosten.*

**Woche 4 (10.–16. März): Phase 2 — Adaptive Allocation (Cloud-Runs)**

| Slot | Aufgabe | GPU-Zeit |
|------|---------|----------|
| Mo Nacht | Phase 2: TextWorld (50 Inst. × 6 Strategien × 5 Runs) | ~10h |
| Di Nacht | Phase 2: Delayed-Cue (50 Inst. × 6 Strategien × 5 Runs) | ~10h |
| Mi Nacht | Buffer: Reruns für fehlerhafte Episoden | ~5h |
| Do | Alle Core-Daten herunterladen + sichern (Home Server) | Lokal |
| Fr–So | Core-Analyse: Mixed-Effects-Modelle, Visualisierungen | Lokal |

**Woche 5 (17.–23. März): Extensions + Analyse**

| Slot | Aufgabe | GPU-Zeit |
|------|---------|----------|
| Mo–Di Nacht | Extension: SC-Signal und/oder Phi-3.5-mini (nach Ergebnis-Sichtung) | ~20–40h |
| Mi–Do | Erweiterte Analyse, Temporal-Degradation-Plots | Lokal |
| Fr–So | Alle Visualisierungen finalisieren, Tabellen erstellen | Lokal |

**Woche 6 (24.–30. März): Schreiben**

| Tag | Kapitel |
|-----|---------|
| Mo | Methodik (4.1–4.6) |
| Di | Methodik (4.7–4.9) + Ergebnisse Phase 1 (5.1–5.4) |
| Mi | Ergebnisse Phase 2 (6.1–6.4) |
| Do | Diskussion (7.1–7.6) |
| Fr | Introduction + Conclusion |
| Sa | Theoretischer Hintergrund (Kernteile) |
| So | Revision, Formatierung, Literaturverzeichnis |

*Theoretischer Hintergrund und Related Work werden parallel während der Experimentwochen geschrieben — immer wenn GPU-Runs laufen und Wartezeit entsteht.*

---

## V. Feasibility-Pilotstudie

### Ziel

Die Pilotstudie beantwortet in ~2 GPU-Stunden und mit ~$0.44 eine einzige Frage: **Funktioniert die gesamte Pipeline End-to-End, und stimmen die Compute-Schätzungen?**

### Vorbereitung (lokal auf M1, bevor Cloud-Geld fließt)

1. GitHub-Repository anlegen (`thesis-metacognitive-allocation`, private)
2. Pilot-Skript lokal vorbereiten (Pseudo-Code, Config, `requirements.txt`)
3. `setup_cloud.sh` schreiben (automatisierte Pod-Einrichtung)

### Pilot-Ablauf (Cloud, ~2 Stunden)

**Setup (~20 min, $0.07):**

```bash
# RunPod Pod starten: RTX 3090, PyTorch 2.x Template, 30GB Volume
# SSH-Verbindung herstellen
pip install vllm transformers textworld numpy pandas scipy
# Qwen2.5-3B-Instruct auf Volume laden (einmalig, persistent)
```

**Test 1 — Inferenzgeschwindigkeit (~10 min, $0.04):**

```python
# 50 Prompts unterschiedlicher Länge
# Messen: tok/s, Latenz pro Call, VRAM-Nutzung
# Erwartung: 100–150 tok/s
# Fallback bei <80 tok/s: Budget-Schätzung × 1.5 korrigieren
```

**Test 2 — Token-Entropie-Extraktion (~10 min, $0.04):**

```python
# vLLM logprobs-Parameter testen
# Entropie über Output-Tokens berechnen
# Prüfe: Variiert Entropie zwischen "einfachen" und "schweren" Prompts?
# Fallback: HuggingFace Transformers mit output_scores=True
```

**Test 3 — Verbalisierte Konfidenz (~10 min, $0.04):**

```python
# Prompt: "Answer, then rate your confidence 0-100."
# 10 Fragen (5 leicht, 5 schwer)
# Prüfe: Numerische Werte parsebar? Leichte Korrelation mit Korrektheit?
# Fallback: Few-Shot-Beispiele im Prompt
```

**Test 4 — TextWorld Mini-Environment (~20 min, $0.07):**

```python
# 3 Mini-Environments generieren (einfach, 3–5 Räume)
# Agent-Loop: Observation → LM-Call → Action → nächste Observation
# Prüfe: Generiert der Agent valide Aktionen?
# Prüfe: Wie viele Steps bis Completion/Failure?
# Fallback bei Installation-Problemen: Eigene Text-Environments
```

**Test 5 — End-to-End Mini-Experiment (~40 min, $0.15):**

```python
# 5 TextWorld-Instanzen × 3 Compute-Stufen × 1 Run = 15 Episoden
# Pro Episode:
#   - TLE extrahieren (via Logprobs)
#   - VC extrahieren (via Prompt)
#   - Task-Erfolg + Steps erfassen
#   - LM-Calls + Tokens + Wall-Clock-Time loggen
# Speichern: Strukturiertes JSON pro Episode
```

Erwartetes Ergebnis: 15 vollständige Datenpunkte → Hochrechnung auf Gesamtexperiment.

**Test 6 — Logging & Download (~10 min, $0.04):**

```python
# JSON-Logs vollständig und parsebar?
# Daten vom Pod downloadbar (rsync/scp)?
# Minimale Analyse-Pipeline: ECE-Berechnung auf 15 Datenpunkten
```

### Pilot-Budget

| Posten | Dauer | Kosten |
|--------|-------|--------|
| Setup + Modell-Download | 20 min | $0.07 |
| Tests 1–3 (Benchmarks) | 30 min | $0.12 |
| Test 4 (TextWorld) | 20 min | $0.07 |
| Test 5 (End-to-End) | 40 min | $0.15 |
| Test 6 + Aufräumen | 10 min | $0.04 |
| **Pilot Gesamt** | **~2 Stunden** | **$0.44 ≈ €0.40** |

*Modell bleibt auf Network Volume — kein Re-Download bei späteren Sessions.*

### Pilot-Checkliste (Go/No-Go)

| # | Frage | Erwartung | Fallback bei "Nein" |
|---|-------|-----------|-------------------|
| 1 | vLLM läuft mit Qwen2.5-3B auf RTX 3090? | Ja | HuggingFace Transformers |
| 2 | Inferenzgeschwindigkeit ≥80 tok/s? | Ja (100+) | Budget × 1.5 korrigieren |
| 3 | Token-Level-Logprobs extrahierbar? | Ja | HF Transformers mit output_scores |
| 4 | Verbalisierte Konfidenz parsebar? | Ja | Few-Shot Prompting |
| 5 | TextWorld installierbar und lauffähig? | Ja | Eigene Text-Environments |
| 6 | Agent generiert valide TextWorld-Aktionen? | Ja | Action-Space-Constraining via Prompt |
| 7 | C2 Self-Consistency (N=3) + Majority Vote funktioniert? | Ja | Konsistenzprüfung statt Majority Vote |
| 8 | End-to-End-Pipeline produziert vollständige Logs? | Ja | Logging-Framework debuggen |
| 9 | Daten-Download + lokale Analyse machbar? | Ja | RunPod Network Volume als Zwischenspeicher |

**Go-Kriterium:** ≥8 von 10 mit "Ja" (oder gelöstem Fallback). Falls ≤6: Design überarbeiten vor Exposé-Versand.

### Pilot-Outputs

Am Ende des Pilot-Tages:

1. **`pilot_benchmark.json`** — Inferenzgeschwindigkeit, VRAM, Kosten pro Episode
2. **`pilot_calibration.json`** — 15 Datenpunkte mit TLE, VC, Korrektheit, Compute-Stufe
3. **`pilot_cost_validation.md`** — Ist-vs-Soll der Compute-Schätzung
4. **`pilot_feasibility_report.md`** — Go/No-Go-Checkliste mit Ergebnissen
5. **Funktionierender Code** — Agent-Loop, Signal-Extraktion, Logging (bereit für Skalierung)

---

## VI. Technische Architektur

### Code-Struktur

```
thesis-metacognitive-allocation/
├── configs/
│   ├── experiment_core.yaml       # Core-Experiment-Parameter
│   ├── experiment_ext.yaml        # Extensions
│   └── pilot.yaml                 # Pilot-Konfiguration
├── src/
│   ├── agent/
│   │   ├── base_agent.py          # Minimaler Agent-Loop (kein Framework)
│   │   ├── compute_stages.py      # C0, C1 (CoT+Verify), C2 (Self-Consistency / majority vote)
│   │   └── allocator.py           # Regelbasierter Allokator + Baselines
│   ├── signals/
│   │   ├── token_entropy.py       # TLE via vLLM-Logprobs
│   │   ├── verbalized_confidence.py  # VC-Extraktion + Parsing
│   │   └── semantic_consistency.py   # SC (Extension)
│   ├── environments/
│   │   ├── textworld_env.py       # TextWorld-Wrapper
│   │   ├── delayed_cue.py         # Delayed-Cue-Task-Generator
│   │   └── logical_reasoning.py   # Logic-Puzzles (Extension)
│   ├── analysis/
│   │   ├── calibration.py         # ECE, Brier, Reliability Diagrams
│   │   ├── comparison.py          # Mixed-Effects-Modelle
│   │   └── visualization.py       # Plots + Tabellen
│   └── utils/
│       ├── logging_utils.py       # Strukturiertes JSON-Logging
│       ├── model_wrapper.py       # vLLM-Wrapper (Fallback: HF Transformers)
│       └── checkpointing.py       # Episode-Level Checkpointing
├── scripts/
│   ├── run_pilot.py               # Pilotstudie
│   ├── run_phase1.py              # Phase 1 Runner
│   ├── run_phase2.py              # Phase 2 Runner
│   └── setup_cloud.sh             # Pod-Setup (automatisiert)
├── data/
│   ├── tasks/                     # Generierte Task-Instanzen
│   └── results/                   # Experiment-Ergebnisse (JSON pro Episode)
└── requirements.txt
```

### Kritische technische Entscheidungen

**vLLM als Inferenz-Backend:**
vLLM unterstützt den `logprobs`-Parameter nativ — damit ist TLE-Extraktion ohne zusätzlichen Code möglich. ~2–3× schneller als HuggingFace Transformers durch Continuous Batching. Fallback auf HF Transformers mit `output_scores=True` falls nötig.

**Checkpointing pro Episode:**
Jede Episode wird sofort nach Completion als einzelne JSON-Datei gespeichert (`results/phase1/ep_{domain}_{instance}_{stage}_{run}.json`). Bei Pod-Crash geht maximal 1 Episode verloren. Das Script prüft beim Start, welche Episoden existieren, und setzt fort.

**Nacht-Run-Automatisierung:**
```bash
# Abends starten:
python scripts/run_phase1.py --config configs/experiment_core.yaml \
    --resume --checkpoint-dir /workspace/results/phase1/
# Script beendet sich nach Completion → Pod idle → Auto-Stop nach 10min
```

**Kein Agent-Framework:**
Minimaler Agent-Loop in reinem Python — kein LangChain, kein LlamaIndex, keine externen Abhängigkeiten. Jede Compute-Stufe ist eine klar definierte Funktion. Das hält den Code auditierbar und debuggbar.

---

## VII. Risiko-Matrix

| Risiko | Wahrsch. | Impact | Mitigation |
|--------|---------|--------|-----------|
| vLLM-Logprob-Extraktion scheitert | Niedrig | Mittel | HF Transformers Fallback |
| TextWorld-Installation scheitert | Mittel | Mittel | Eigene Text-Environments (simpler) |
| Pod-Crash während Nacht-Run | Mittel | Niedrig | Checkpointing pro Episode |
| Modell generiert keine validen Aktionen | Mittel | Hoch | Few-Shot + Action-Space-Constraining |
| TLE korreliert nicht mit Korrektheit | Mittel | Hoch | Pilot prüft dies; VC als Primär-Signal |
| VC nicht parsebar | Niedrig | Mittel | Robuster Parser + Few-Shot |
| Qwen2.5-3B zu schwach für Tasks | Niedrig–Mittel | Hoch | Pilot validiert; Fallback Phi-3.5-mini |
| Ergebnisse zeigen keinen Effekt | Mittel | Mittel | Null-Ergebnis publizierbar ("SLMs lack metacognition in agent settings") |
| Compute-Budget reicht nicht | Sehr niedrig | Hoch | Core benötigt nur ~$13; $10 bereits geladen |

---

## VIII. Entscheidungsbaum nach Pilotstudie

```
Pilot abgeschlossen
│
├─ ≥8 Checks bestanden (oder Fallbacks gelöst)
│  → GO: Exposé an Professor senden, Woche 2 starten
│
├─ 7 Checks bestanden
│  → GO mit Anpassungen: Problematische Komponente fixen/ersetzen
│
├─ 4–6 Checks bestanden
│  → PAUSE: 1 Woche für technische Problemlösung
│  → Design ggf. vereinfachen (z.B. nur 2 statt 3 Compute-Stufen)
│
└─ <4 Checks bestanden
   → REDESIGN nötig: Alternatives Setup evaluieren (API-basiert?)
```

---

## IX. Zusammenfassung: Die Zahlen auf einen Blick

| Metrik | v1 | **v2** | Δ |
|--------|-----|--------|---|
| Core GPU-Stunden | ~108 | **~59** | −45% |
| Core-Kosten (RunPod 3090) | ~€22 | **~€12** | −45% |
| Full Design + Extensions | ~€81 | **~€33** | −59% |
| Pilot-Kosten | ~€0.60 | **~€0.40** | −33% |
| LM-Calls (Core) | ~185.000 | **~102.500** | −45% |
| Nacht-Sessions (Core) | ~8 | **~5** | −38% |

| Szenario | GPU-h | Kosten | Anteil von $10 Guthaben |
|----------|-------|--------|------------------------|
| Pilot | 2 | $0.44 | 4% |
| Core | 59 | $12.98 | 130% (nachladen: +$5) |
| Core + Extensions | 164 | $36.08 | nachladen: +$30 |

**$10 Startguthaben reicht für Pilot + fast das gesamte Core-Experiment.** Zum Start von Phase 1 (Woche 3) einmalig $5–10 nachladen, dann ist Core abgesichert. Extensions bei Bedarf nachfinanzieren.

---

## X. Nächste Schritte (Diese Woche)

1. ~~RunPod-Account erstellen~~ ✓
2. ~~$10 Guthaben aufladen~~ ✓
3. **GitHub-Repo anlegen** (Private: `thesis-metacognitive-allocation`)
4. **Pilot-Script lokal vorbereiten** (basierend auf Abschnitt V)
5. **Pilot durchführen** (~2 Stunden Cloud-Zeit, ~$0.44)
6. **Go/No-Go-Entscheidung** → Exposé an Professor
