# Redesigned Master Thesis Blueprint (v2 — Refined)

## Keanu Forthmann — M.Sc. Artificial Intelligence, IU Internationale Hochschule

---

# Metacognitive Effort Allocation in Sequential Language Model Agents: Comparing Cognitive Calibration Signals for Adaptive Compute Depth in Multi-Step Decision Tasks

---

## I. Strategische Begründung und Positionierung

### Warum diese Evolution des Originalthemas?

Das ursprüngliche Exposé untersuchte Memory und Planning als *statische* strukturelle Variablen: Für jede Bedingung wurde ein fester Parametersatz definiert und dessen Effekt gemessen. Das war methodisch sauber, aber *deskriptiv* — es beschrieb Performance-Landschaften, ohne ein Prinzip zu identifizieren, *wann welche Konfiguration optimal ist*.

Das redesigned Thesis verschiebt die Frage von "Was passiert bei verschiedenen Memory-Planning-Konfigurationen?" zu "Kann ein Agent *selbst einschätzen*, wann er mehr nachdenken muss — und lassen sich die Signale dafür aus der kognitiven Psychologie ableiten?"

Diese Verschiebung trifft alle drei Prioritäten:

**Prio 1 — DeepMind-Caliber:** DeepMinds einflussreichste Arbeiten (AlphaGo, MuZero, Adaptive Computation Time, PonderNet) teilen ein gemeinsames Muster: Sie formalisieren *wann* und *wie viel* ein System nachdenken sollte als optimierbares Problem. Die vorliegende Arbeit überträgt exakt dieses Prinzip auf LM-Agenten, nutzt aber anstelle gelernter Halting-Mechanismen *kognitionspsychologisch motivierte Signale* als Steuerungsmechanismus.

**Prio 2 — Cutting-Edge Relevanz:** Test-Time Compute Scaling ist *das* zentrale Thema der AI-Forschung 2025/2026. OpenAIs o1/o3-Modelle, DeepSeek-R1, und Googles Gemini-Thinking haben gezeigt, dass mehr Inferenz-Compute massive Leistungssteigerungen ermöglicht. Jüngste Arbeiten wie EAGer (Scalena et al., 2025) und MeCo (Li et al., 2025) haben begonnen, metacognitive Signale für Compute-Allokation in *Single-Turn-Settings* zu nutzen. Die offene Frage — die diese Arbeit adressiert — ist: *Funktionieren diese Mechanismen auch in sequentiellen Agent-Settings, wo Allokationsentscheidungen kumulieren und interagieren?*

**Prio 3 — Psychologie-USP:** Die Arbeit ist in ihrer theoretischen Architektur eine *kognitionspsychologische Arbeit, die mit AI-Methoden implementiert wird*. Dual-Process Theory (Kahneman), Metacognition (Nelson & Narens), Expected Value of Control (Shenhav et al.) und Confidence Calibration sind die *generative Theorie*, aus der die experimentellen Hypothesen abgeleitet werden.

### Abgrenzung gegen die existierende Literatur

Jüngste Arbeiten haben das Konzept metacognitiver Signale für Compute-Allokation etabliert:

- **EAGer** (Scalena et al., 2025) nutzt Token-Level-Entropie, um innerhalb eines *einzelnen Generierungsdurchlaufs* zu entscheiden, wo neue Reasoning-Branches starten. Die Granularität ist Token-Level, das Setting Single-Turn, und es wird nur ein Signal verwendet.
- **MeCo** (Li et al., 2025) trainiert Probes auf *internen Repräsentationen* (Hidden States), um zu entscheiden, ob ein Tool aufgerufen wird. Die Entscheidung ist binär (Tool ja/nein), und es wird Zugang zu internen Modellzuständen benötigt (White-Box).
- **Li Ji-An et al. (2025)** und das **MGV-Framework** (2025) liefern Grundlagenarbeit zur Frage, ob LLMs metacognitive Monitoring-Fähigkeiten besitzen, ohne Anwendung auf Compute-Allokation.

Die vorliegende Arbeit erweitert dieses Fundament in eine präzise definierte Richtung:

> "While recent work has demonstrated that metacognitive signals such as token entropy (EAGer; Scalena et al., 2025) and representation-based probes (MeCo; Li et al., 2025) can guide compute allocation in single-turn settings, no systematic investigation exists for sequential agent decision-making — a setting where allocation decisions compound over multiple steps, interact with memory constraints, and may exhibit temporal degradation patterns predicted by cognitive psychology. This thesis provides the first controlled comparison of metacognitive calibration signals for adaptive compute depth allocation in multi-step language model agents."

---

## II. Theoretischer Rahmen

### A. Das Compute-Effort-Problem in LM-Agenten

Wenn ein LM-Agent eine Sequenz von Entscheidungen treffen muss, steht er bei jedem Schritt vor einer impliziten Ressourcen-Allokationsfrage: Soll das System direkt antworten (niedrige Compute-Kosten, schnell, aber fehleranfällig) oder soll es erst deliberieren (Chain-of-Thought, Self-Verification — hohe Compute-Kosten, langsamer, aber potenziell akkurater)?

Aktuelle Systeme lösen dieses Problem auf drei unbefriedigende Weisen:

1. **Fixe Budgets:** Jede Anfrage bekommt dasselbe Compute-Budget. Ineffizient, weil triviale Entscheidungen überkompensiert und schwierige unterkompensiert werden.

2. **Trainierte interne Mechanismen:** o1/o3-artige Modelle lernen intern, wann sie länger "nachdenken" sollen. Das funktioniert, ist aber (a) nicht interpretierbar, (b) nicht übertragbar zwischen Modellen, und (c) erfordert spezialisiertes Training.

3. **Prompt-Level-Allokation:** Arbeiten wie EAGer entscheiden *pro Prompt*, wie viel Compute investiert wird. In Agent-Settings, wo ein Prompt aus vielen sequentiellen Entscheidungsschritten besteht, ist diese Granularität zu grob — verschiedene Schritte innerhalb derselben Episode können radikal unterschiedliche Schwierigkeitsgrade haben.

### B. Kognitionspsychologische Grundlagen

Drei Theorietraditionen sind direkt relevant:

**Dual-Process Theory (Kahneman, 2011; Evans & Stanovich, 2013):**
Menschliches Denken operiert in zwei Modi — System 1 (schnell, automatisch, heuristisch) und System 2 (langsam, kontrolliert, deliberativ). Der Wechsel zwischen den Systemen wird durch *Überraschung*, *wahrgenommene Schwierigkeit* und *Fehlererwartung* ausgelöst.

**Metacognition und Monitoring (Nelson & Narens, 1990; Flavell, 1979):**
Metacognition bezeichnet die Fähigkeit, das eigene Denken zu überwachen und zu steuern. Zentrale Signale sind Feeling of Knowing (FOK), Judgment of Learning (JOL) und Confidence Calibration. Diese Signale sind *informativ aber imperfekt* — Menschen nutzen heuristische Cues (Fluency, Familiarity, Coherence) als Proxy für tatsächliche Korrektheit. Entscheidend: Metacognitive Monitoring-Genauigkeit verschlechtert sich unter erhöhter kognitiver Last (Efklides, 2006) — eine Vorhersage, die direkt auf das agentische Setting übertragbar ist.

**Expected Value of Control (EVC) Theory (Shenhav, Botvinick & Cohen, 2013):**
Das EVC-Framework formalisiert, wie das Gehirn entscheidet, *wie viel kognitive Kontrolle* in eine Aufgabe investiert wird. Die optimale Kontrollintensität maximiert:

    EVC(signal, intensity) = E[Reward(signal, intensity)] − Cost(intensity)

### C. Die Brücke: Metacognitive Signale als Compute-Allokator im Agent-Setting

Die zentrale theoretische Innovation besteht darin, metacognitive Signale als *Step-Level-Steuerungssignale für Compute-Tiefe* in sequentiellen LM-Agenten zu operationalisieren.

Der entscheidende Unterschied zu existierenden Ansätzen: Im agentischen Setting kommen Faktoren hinzu, die in Single-Turn-Settings nicht existieren:

1. **Kumulative Allokation:** Falsche Allokation bei Schritt 5 kann die Informationsbasis für Schritt 10 ruinieren.
2. **Memory-Abhängigkeit:** Die Qualität metacognitiver Signale kann sich über eine Episode verändern, weil der Kontext (History) wächst.
3. **Temporal Degradation:** Aus der Psychologie ist bekannt, dass Metacognitive Monitoring unter kognitiver Last an Genauigkeit verliert — das wachsende Context-Fenster im Agenten ist das computationale Äquivalent.

Die Analogie:
- **System 1** = Direkte Inferenz (1 Forward Pass)
- **System 2** = Deliberation (CoT + Self-Verification)
- **Metacognitive Signale** = Entscheidungsmechanismus für den Wechsel

---

## III. Forschungsfragen und Hypothesen

### Forschungsfragen

**RQ1 (Signalqualität):** Wie gut sind metacognitive Proxy-Signale (Token-Level Entropy, Verbalisierte Konfidenz) in Small Language Models kalibriert — d.h. in welchem Maße korrelieren sie mit tatsächlicher Entscheidungskorrektheit in Multi-Step-Agent-Tasks?

**RQ2 (Adaptive Allokation):** Führt eine auf metacognitiven Signalen basierende adaptive Step-Level-Compute-Allokation zu besseren Performance/Compute-Trade-offs als fixe Allokationsstrategien?

**RQ3 (Temporale Degradation):** Verändert sich die Kalibrierungsqualität metacognitiver Signale im Verlauf einer Episode — d.h. werden sie mit zunehmendem Context weniger prädiktiv?

**RQ4 (Cross-Domain Stability):** Sind die identifizierten Signal-Performance-Relationen domänenspezifisch oder transferieren sie über verschiedene Aufgabentypen?

### Gerichtete Hypothesen

**H1 (Calibration Hypothesis):** Token-Level Entropy ist ein besser kalibrierter metacognitiver Proxy als verbalisierte Konfidenz, da letztere durch RLHF-Training systematisch überoptimistisch verzerrt ist.
*Psychologische Grundlage:* Overconfidence-Bias in metakognitiven Urteilen (Fischhoff et al., 1977); in LMs verstärkt durch Reinforcement Learning from Human Feedback.

**H2 (Adaptive Superiority Hypothesis):** Adaptive Compute-Allokation erreicht ≥90% der Performance der Always-Deliberate-Strategie bei ≤50% der Compute-Kosten, weil ein substantieller Anteil der Entscheidungsschritte trivial ist und korrekt von System-1-Inferenz gelöst werden kann.
*Psychologische Grundlage:* Pareto-Verteilung kognitiver Anforderungen in natürlichen Umgebungen (Anderson & Schooler, 1991).

**H3 (Temporal Degradation Hypothesis):** Die Kalibrierung metacognitiver Signale nimmt im Verlauf einer Episode ab. Bei späteren Entscheidungsschritten (mit längerem Context-Fenster) ist Token-Level-Entropie weniger prädiktiv für tatsächliche Korrektheit als bei frühen Schritten.
*Psychologische Grundlage:* Metacognitive Monitoring Accuracy sinkt unter erhöhter kognitiver Last (Efklides, 2006). Der wachsende Prompt-Context ist das LM-Äquivalent kognitiver Auslastung.

**H4 (Domain Modulation Hypothesis):** Metacognitive Signale sind bei gedächtnisintensiven Aufgaben (Delayed Recall) schlechter kalibriert als bei Navigation-Aufgaben, da Memory-Fehler in LMs andere Token-Entropie-Signaturen produzieren als Planungsfehler.
*Psychologische Grundlage:* Domänenspezifität von FOK-Accuracy (Schwartz & Metcalfe, 2011).

---

## IV. Methodik

### A. Überblick: Vereinfachtes experimentelles Design

Die Studie implementiert ein kontrolliertes Experiment mit einem **2 × 3 × 2 Core-Design**:

- **Faktor 1 — Metacognitives Signal (2 Stufen):** Token-Level Entropy (TLE), Verbalisierte Konfidenz (VC)
- **Faktor 2 — Compute-Stufe (3 Stufen):** Direct Inference (C0), Chain-of-Thought mit Self-Check (C1), Best-of-N Sampling (C2)
- **Faktor 3 — Task-Domäne (2 Stufen):** Text-Navigation, Delayed-Cue Recall

**Vereinfachungen gegenüber v1 und Begründung:**

| v1 | v2 | Begründung |
|----|-----|-----------|
| 4 Compute-Stufen (Direct, Short CoT, Deep CoT, Tree Search) | 3 Stufen (Direct, CoT+Verify, Best-of-N) | CoT-Länge ist bei Instruction-Tuned-Modellen nicht extern steuerbar; Tree Search ersetzt durch Best-of-N (einfacher, dennoch starke Baseline nach Snell et al., 2024) |
| 3 Signale (TLE, VC, SC) | 2 Signale Core (TLE, VC) | Semantic Consistency erfordert 5× Sampling pro Step — zu teuer für Core; bleibt Extension |
| 3 Domänen | 2 Domänen Core | Logical Reasoning bleibt Extension; 2 Domänen reichen für Cross-Domain-Test (RQ4) |
| 2 Modelle | 1 Modell Core | Zweites Modell bleibt Extension; 1 Modell reicht für Kernhypothesen |
| 30 Runs pro Bedingung | 5 Runs | Bei Temperature 0.3 ist Varianz gering; 5 Runs liefern stabile Mittelwerte |

Das Design hat zwei Phasen:

**Phase 1 — Calibration Mapping (RQ1, RQ3):** Für jede Kombination von Signal und Domäne wird die Kalibriertheit gemessen. Der Agent bearbeitet Aufgaben, wobei alle Compute-Stufen durchlaufen werden. Die Korrelation zwischen Signal und Korrektheit wird quantifiziert, aufgeschlüsselt nach Episode-Position (für RQ3).

**Phase 2 — Adaptive Allocation (RQ2, RQ4):** Die metacognitiven Signale steuern einen regelbasierten Allokator. Die Performance wird gegen Baselines verglichen.

### B. Operationalisierung der Metacognitiven Signale

**Signal 1 — Token-Level Entropy (TLE):**
Für den initialen Forward Pass (System-1-Response) wird die Entropie der Token-Verteilung über die Antwort-Tokens berechnet:

    H(t) = -Σ p(x_i) log p(x_i)

Aggregation über die gesamte Antwort via Mean und Max. Hohe Entropie signalisiert Unsicherheit.

*Psychologische Analogie:* Processing Fluency als metacognitives Cue (Alter & Oppenheimer, 2009). Hohe Token-Entropie ist das computationale Äquivalent niedriger kognitiver Fluency.

*Abgrenzung zu EAGer:* EAGer nutzt Token-Entropie *innerhalb* eines Generierungsdurchlaufs, um zu entscheiden, wo neue Branches starten (Token-Level, Single-Turn). Wir nutzen die Entropie der *gesamten Step-Antwort* als aggregiertes Signal für die Compute-Stufe des *nächsten* Steps (Step-Level, sequentiell). Verschiedene Granularität, verschiedener Entscheidungsgegenstand.

**Signal 2 — Verbalisierte Konfidenz (VC):**
Das Modell wird explizit aufgefordert, nach seiner initialen Antwort eine Konfidenz-Einschätzung abzugeben ("Rate your confidence in this decision from 0 to 100"). Die verbalisierte Konfidenz wird als numerischer Wert extrahiert.

*Psychologische Analogie:* Feeling of Knowing (FOK) — ein explizites metacognitives Urteil (Koriat, 1993). FOK-Urteile sind informativ, aber systematisch verzerrt.

*Abgrenzung zu MeCo:* MeCo nutzt trainierte Probes auf Hidden-Layer-Aktivierungen (White-Box). Verbalisierte Konfidenz ist ein reines Output-Level-Signal (Gray-Box), das keinen Zugang zu Modell-Internals erfordert und somit modell-agnostisch einsetzbar ist.

### C. Compute-Stufen (Deliberationsintensität)

| Stufe | Beschreibung | Analogie | LM-Calls/Step |
|-------|-------------|----------|---------------|
| C0 — Direct | Einzelner Forward Pass, keine Elaboration | System 1 | 1 |
| C1 — CoT + Verify | Chain-of-Thought-Generierung + Self-Verification-Prompt | Deliberation | 2 |
| C2 — Best-of-N | N=3 parallele Generierungen + Majority Vote | Exhaustive Analyse | 3 |

**Begründung der Vereinfachung:** Die klare Abstufung 1/2/3 LM-Calls pro Step macht den Compute-Kontrast quantifizierbar und reproduzierbar. Best-of-N ersetzt Tree Search: einfacher zu implementieren, keine Branching-Logik nötig, und in der aktuellen Literatur als starke Test-Time-Compute-Methode etabliert (Snell et al., 2024).

### D. Adaptiver Allokator (Regelbasiert)

Der Allokator nutzt das metacognitive Signal s ∈ [0,1] (normalisiert) und zwei Schwellenwerte θ₁ < θ₂:

    if s < θ₁:        → C0 (Direct — hohe Konfidenz)
    elif s < θ₂:      → C1 (CoT + Verify — moderate Unsicherheit)
    else:              → C2 (Best-of-N — hohe Unsicherheit)

Die Schwellenwerte werden auf einem Validierungssplit (10% der Aufgaben pro Domäne) über Grid-Search optimiert.

### E. Task-Domänen

**Domäne 1 — Text-Navigation (TextWorld):**
Standardisierte TextWorld-Instanzen mit kontrollierter Schwierigkeit. Der Agent muss ein Zielobjekt finden und eine Sequenz von Aktionen ausführen. Episoden haben typischerweise 8–15 Steps. Testet primär sequentielle Planung mit räumlichem Memory.

**Domäne 2 — Delayed-Cue Recall:**
Aufgaben, bei denen kritische Information früh gegeben wird, gefolgt von Distraktoren, und die Information erst spät benötigt wird. Direkte Operationalisierung des Delayed-Match-to-Sample-Paradigmas aus der kognitiven Psychologie. Testet primär, ob metacognitive Signale Memory-Fehler erkennen — eine Fähigkeit, die in der Single-Turn-Literatur nicht adressiert wird.

### F. Modellauswahl

**Core:** Qwen2.5-3B-Instruct (oder funktional äquivalentes Modell zum Zeitpunkt der Durchführung)

**Extension:** Phi-3.5-mini-instruct (3.8B) als zweites Modell für Generalisierbarkeitstest.

Begründung für SLMs: (a) Lokale Inferenz ohne API-Abhängigkeit, (b) Reproduzierbarkeit, (c) Ressourcen-Constraints machen adaptive Allokation *relevanter* als bei großen Modellen, (d) Edge-Deployment-Implikationen.

### G. Abhängige Variablen

**Primär:**
- Task Success Rate (binär pro Episode)
- Normalized Compute Cost (Anzahl LM-Calls pro Episode)
- Efficiency Score: Success Rate / Normalized Compute Cost

**Sekundär:**
- Expected Calibration Error (ECE) der metacognitiven Signale
- Brier Score für Signalqualität
- Per-Step Allocation Distribution (wie oft wird C0/C1/C2 gewählt?)
- Signal Calibration by Episode Position (ECE aufgeschlüsselt nach Step-Index, für RQ3)

### H. Baselines

| Baseline | Beschreibung | Funktion |
|----------|-------------|----------|
| Always-C0 | Immer Direct Inference | Lower Bound (Speed) |
| Always-C2 | Immer Best-of-N | Upper Bound (Performance) |
| Random-Alloc | Zufällige Compute-Stufe pro Step | Kontrolle für den Effekt *informierter* Allokation |
| EAGer-Style | Episoden-Level-Allokation: Entropie des ersten Steps bestimmt ein festes Compute-Level für alle Steps der Episode | Kontrolle für den Mehrwert von *Step-Level*- vs. Prompt-Level-Allokation |

Die EAGer-Style-Baseline testet die zentrale Abgrenzung: Wenn Step-Level-Allokation besser abschneidet als Episoden-Level-Allokation, belegt das den Mehrwert der feineren Granularität — und damit den Beitrag dieser Arbeit gegenüber existierenden Ansätzen.

### I. Statistische Analysestrategie

**Für RQ1 (Kalibrierung):**
- Expected Calibration Error (ECE) und Brier Scores für TLE und VC pro Domäne
- Reliability Diagrams (Calibration Plots)
- Vergleich TLE vs. VC via Permutationstest auf ECE-Differenzen

**Für RQ2 (Adaptive Superiority):**
- Mixed-Effects-Modell: `Success ~ Strategy × Domain + (1|Task_Instance)`
- Pairwise Contrasts (Tukey-adjusted) zwischen adaptiver Strategie und Baselines
- Effektstärken (Cohen's d für Performance, Ratio für Compute)

**Für RQ3 (Temporal Degradation):**
- `Signal_ECE ~ Step_Position × Domain + (1|Task_Instance)`
- Visualisierung: Calibration Curve frühe Steps vs. späte Steps
- *Kostet null zusätzliche GPU-Stunden* — Daten fallen in Phase 1 ohnehin an

**Für RQ4 (Cross-Domain):**
- Signalqualität × Domäne Interaktion im Mixed-Effects-Modell
- Deskriptiver Vergleich der Allokator-Schwellenwerte zwischen Domänen

### J. Sample Size und Power

- 50 Task-Instanzen pro Domäne (2 Domänen = 100 Instanzen)
- 5 Wiederholungen pro Instanz pro Bedingung (Temperature 0.3)
- 1 Modell im Core

Für den primären Vergleich (H2: Adaptive-TLE vs. Always-C2) mit erwarteter Effektstärke d = 0.5 ergibt eine Power-Analyse (α = .05, Power = .80) ~34 Instanzen pro Bedingung. Mit 50 Instanzen pro Domäne ist die Studie ausreichend gepowert.

---

## V. Priorisierungsstrategie (Core vs. Extension)

### Core (Muss für Abgabe vorhanden sein)

1. Signal-Kalibrierungsanalyse für TLE und VC auf Text-Navigation und Delayed-Cue Recall mit 1 Modell
2. Adaptiver Allokator (TLE-basiert und VC-basiert) vs. 4 Baselines
3. Temporal Degradation-Analyse (RQ3)
4. Cross-Domain-Vergleich (RQ4) zwischen den zwei Core-Domänen

**→ Core-Designraum: 2 × 3 × 2 = 12 Bedingungen (Phase 1), 6 Strategien × 2 Domänen (Phase 2). Kompakt und machbar in ~108 GPU-Stunden (~€22).**

### Extension A (Wenn Zeit vorhanden)

5. Semantic Consistency als drittes Signal
6. Zweites Modell (Phi-3.5-mini)

### Extension B (Bonus)

7. Logical Reasoning als dritte Domäne
8. Qualitative Fehleranalyse (Fallstudien)

---

## VI. Zeitplan (6 Wochen Experiment + Schreiben parallel)

| Woche | Phase | Deliverable |
|-------|-------|-------------|
| 1 | Pilot + Setup | Feasibility-Pilot auf Cloud-GPU, Codebase, Task-Generierung |
| 2 | Implementation | Agent-Loop, Signal-Extraktion, Compute-Stufen, lokale Tests |
| 3 | Phase 1 (Cloud) | Calibration Mapping: TLE + VC auf 2 Domänen, 1 Modell |
| 4 | Phase 2 (Cloud) | Adaptive Allocation Experiment + Baselines |
| 5 | Extensions + Analyse | Optionale Erweiterungen; Core-Analyse, Visualisierungen |
| 6 | Analyse + Schreiben | Erweiterte Auswertung, Ergebniskapitel, Diskussion |

*Theorie-Kapitel und Methodik werden parallel während der Experimentwochen geschrieben.*

---

## VII. Vorläufige Gliederung

1. **Introduction**
   1.1 The Compute Allocation Problem in Language Model Agents
   1.2 Research Gap: From Single-Turn to Sequential Settings
   1.3 Research Questions and Hypotheses
   1.4 Structure of the Thesis

2. **Theoretical Background**
   2.1 Dual-Process Theory and Cognitive Effort Allocation
       2.1.1 System 1 and System 2: Definitions and Empirical Evidence
       2.1.2 Expected Value of Control Theory
       2.1.3 Metacognitive Monitoring and Control
       2.1.4 Temporal Effects on Metacognitive Accuracy
   2.2 Language Models as Sequential Decision Systems
       2.2.1 Transformer Architecture and Token-Level Inference
       2.2.2 Chain-of-Thought and Deliberative Reasoning
       2.2.3 Test-Time Compute Scaling: Current Approaches
   2.3 Small Language Models: Capabilities and Constraints

3. **Related Work**
   3.1 Entropy-Based Compute Allocation (EAGer, Entropy Adaptive Decoding)
   3.2 Metacognitive Probes and Tool-Use Decisions (MeCo, AutoMeCo)
   3.3 Metacognitive Capabilities in LLMs (Li Ji-An et al., MGV Framework)
   3.4 Confidence Calibration in Language Models
   3.5 Agent Architectures with Variable Reasoning Depth
   3.6 Positioning of the Present Work

4. **Methodology**
   4.1 Research Design Overview
   4.2 Operationalization of Metacognitive Signals
       4.2.1 Token-Level Entropy
       4.2.2 Verbalized Confidence
   4.3 Compute Stages and Deliberation Mechanisms
   4.4 Adaptive Allocation Mechanism
   4.5 Task Environments
       4.5.1 Text-Based Navigation
       4.5.2 Delayed-Cue Recall
   4.6 Baselines (incl. EAGer-Style)
   4.7 Model Selection and Infrastructure
   4.8 Statistical Analysis Plan
   4.9 Methodological Limitations

5. **Results: Signal Calibration Analysis (Phase 1)**
   5.1 Overall Calibration of Token-Level Entropy
   5.2 Overall Calibration of Verbalized Confidence
   5.3 Temporal Degradation Across Episode Steps
   5.4 Cross-Domain Comparison of Signal Quality

6. **Results: Adaptive Allocation Experiments (Phase 2)**
   6.1 Performance Comparison Against Baselines
   6.2 Compute Efficiency Analysis
   6.3 Step-Level vs. Prompt-Level Allocation (EAGer-Style Comparison)
   6.4 Allocation Patterns: When Does the Agent Choose to Deliberate?

7. **Discussion**
   7.1 Interpretation Through the Lens of Dual-Process Theory
   7.2 Metacognitive Signal Quality in Sequential vs. Single-Turn Settings
   7.3 Temporal Degradation: What Growing Context Does to Self-Monitoring
   7.4 Implications for Efficient Agent Design
   7.5 Limitations
   7.6 Ethical Considerations

8. **Conclusion and Future Work**
   8.1 Summary of Contributions
   8.2 Directions for Future Research
   8.3 Toward Metacognition-Aware AI Agents

---

## VIII. Vorläufige Literaturliste (Erweitert mit 2025-Quellen)

### Kognitive Psychologie und Metacognition
- Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.
- Evans, J. S. B. T., & Stanovich, K. E. (2013). Dual-process theories of higher cognition. Perspectives on Psychological Science, 8(3), 223–241.
- Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013). The expected value of control. Neuron, 79(2), 217–240.
- Nelson, T. O., & Narens, L. (1990). Metamemory: A theoretical framework and new findings. Psychology of Learning and Motivation, 26, 125–173.
- Koriat, A. (1993). How do we know that we know? The accessibility model of the feeling of knowing. Psychological Review, 100(4), 609–639.
- Koriat, A. (2012). The self-consistency model of subjective confidence. Psychological Review, 119(1), 80–113.
- Hart, J. T. (1965). Memory and the feeling-of-knowing experience. Journal of Educational Psychology, 56(4), 208–216.
- Flavell, J. H. (1979). Metacognition and cognitive monitoring. American Psychologist, 34(10), 906–911.
- Fischhoff, B., Slovic, P., & Lichtenstein, S. (1977). Knowing with certainty. Journal of Experimental Psychology: Human Perception and Performance, 3(4), 552–564.
- Alter, A. L., & Oppenheimer, D. M. (2009). Uniting the tribes of fluency to form a metacognitive nation. Personality and Social Psychology Review, 13(3), 219–235.
- Schwartz, B. L., & Metcalfe, J. (2011). Tip-of-the-tongue (TOT) states. Memory & Cognition, 39(5), 737–749.
- Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory. Psychological Science, 2(6), 396–408.
- Efklides, A. (2006). Metacognition and affect: What can metacognitive experiences tell us about the learning process? Educational Research Review, 1(1), 3–14.
- Baddeley, A. (2000). The episodic buffer. Trends in Cognitive Sciences, 4(11), 417–423.

### Metacognition in LLMs (2025 — Direkte Vorgängerarbeiten)
- Scalena, D., Zotos, L., Fersini, E., Nissim, M., & Üstün, A. (2025). EAGER: Entropy-aware generation for adaptive inference-time scaling. arXiv:2510.11170.
- Li, W., Li, D., Dong, K., Zhang, C., et al. (2025). Adaptive tool use in large language models with meta-cognition trigger. ACL 2025.
- Li Ji-An et al. (2025). Language models are capable of metacognitive monitoring and control of their internal activations. arXiv preprint.
- Ma, Y., et al. (2025). Large language models have intrinsic meta-cognition, but need a good lens (AutoMeCo). EMNLP 2025.
- Steyvers, M., & Peters, M. (2025). Metacognitive capabilities in large language models. Nature Reviews Psychology.
- Monitor-Generate-Verify Framework (2025). Formalising metacognitive theory for LLMs. arXiv preprint.

### Test-Time Compute und Adaptive Inference
- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv:2408.03314.
- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv:1603.08983.
- Banino, A., et al. (2022). PonderNet: Learning to ponder. ICML 2022.
- Schuster, T., et al. (2022). Confident adaptive language modeling. NeurIPS 2022.
- Fu, Y., et al. (2025). DeepConf: Deep think with confidence. arXiv preprint.

### LM-Reasoning und Chain-of-Thought
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS 2022.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate problem solving with large language models. NeurIPS 2023.
- Shinn, N., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. NeurIPS 2023.
- Madaan, A., et al. (2023). Self-Refine: Iterative refinement with self-feedback. NeurIPS 2023.

### Confidence Calibration in Language Models
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? ICLR 2024.
- Guo, C., et al. (2017). On calibration of modern neural networks. ICML 2017.
- Lin, S., Hilton, J., & Evans, O. (2022). Teaching models to express their uncertainty in words. TMLR 2022.

### LM-Agenten und Sequentielle Entscheidungssysteme
- Yao, S., et al. (2023). ReAct: Synergizing reasoning and acting in language models. ICLR 2023.
- Zhou, A., et al. (2024). Language Agent Tree Search unifies reasoning, acting, and planning. ICML 2024.

### Small Language Models
- Abdin, M., et al. (2024). Phi-3 Technical Report. arXiv:2404.14219.
- Yang, A., et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.

### Environments und Benchmarks
- Côté, M.-A., et al. (2018). TextWorld: A learning environment for text-based games. AAAI 2018.
- Hausknecht, M., et al. (2020). Interactive fiction games: A colossal adventure. AAAI 2020.

### Reinforcement Learning Foundations
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
- Silver, D., et al. (2016). Mastering the game of Go. Nature, 529, 484–489.
- Schrittwieser, J., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588, 604–609.

---

## IX. Warum dieses Design die Schwächen des Originals behebt

| Schwäche im Original | Lösung im Redesign v2 |
|---------------------|----------------------|
| Fehlende statistische Analysestrategie | Mixed-Effects-Modelle, ECE, Brier Scores, Permutationstests |
| Keine Power-Analyse | 50 Instanzen/Domäne, 5 Runs, Power >.80 für d=0.5 |
| "Regime Transition" vage definiert | Ersetzt durch "Temporal Degradation" (ECE als Funktion der Step-Position) |
| Keine gerichteten Hypothesen | 4 Hypothesen mit psychologischer Herleitung |
| Feasibility-Risiko | Kompaktes 2×3×2 Core-Design; ~108 GPU-Stunden, ~€22 |
| "Hyperparameter-Tuning"-Angriffsfläche | Theoriegeleitete Hypothesen + explizite Abgrenzung zu EAGer/MeCo |
| Bounded-Rationality-Brücke dünn | EVC + Dual-Process Theory + Temporal Degradation aus Metacognitions-Literatur |
| Fehlende Abgrenzung gegen existierende Literatur | EAGer, MeCo, Li Ji-An, MGV explizit eingebettet; Lücke "sequentielle Agents" klar benannt |

---

## X. Positionierung für Research-Karriere

**Workshop-Paper-Potential:** Die Calibration-Analyse mit Temporal-Degradation-Befund wäre als Workshop-Paper einreichbar: "Do Small Language Models Know When They Don't Know? Metacognitive Calibration Across Sequential Agent Decisions."

**Nachfolge-Projekt (PhD):** Die Arbeit legt die Grundlage für ein PhD-Projekt, das den regelbasierten Allokator durch einen *gelernten* Allokator ersetzt — analog zu PonderNet, aber für Agenten, mit psychologisch informierter Reward-Funktion.

**Narrativ für Bewerbungen:** "I conducted the first systematic comparison of metacognitive calibration signals for adaptive compute allocation in sequential language model agents, demonstrating that signal quality degrades over episode length in patterns consistent with human metacognitive load effects — a finding that only emerges in agent settings and has direct implications for efficient AI system design."
