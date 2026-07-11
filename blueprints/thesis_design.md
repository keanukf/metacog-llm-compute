# Redesigned Master Thesis Blueprint (v2 — Refined)

> **Design update (March 2026):** Domain 2 changed from Delayed-Cue Recall to Tower of Hanoi. See `chapters/outline.md` for current design. Rationale: Tower of Hanoi provides a genuinely sequential planning task under full observability, maintaining narrative consistency with the thesis focus on sequential decision-making. H4 now contrasts exploration (TextWorld) vs. planning (Tower of Hanoi) rather than sequential vs. retrieval.

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

*Kapitelstruktur (Kap. 2):* Zuerst das computationale Problem etablieren (LM als sequentielle Entscheidungssysteme, Allokationsmöglichkeit), dann die theoretische Linse aus der Kognitionspsychologie (Dual-Process, EVC, Metacognition, Temporal Degradation), dann die Brücke zu LMs (CoT, Test-Time Compute, LM-Metacognition, Small LMs). So trägt die Psychologie das Gewicht — jede eingeführte Idee mappt auf eine Designentscheidung, die der Leser bereits erwartet.

### A. Das Compute-Effort-Problem in LM-Agenten

Wenn ein LM-Agent eine Sequenz von Entscheidungen treffen muss, steht er bei jedem Schritt vor einer impliziten Ressourcen-Allokationsfrage: Soll das System direkt antworten (niedrige Compute-Kosten, schnell, aber fehleranfällig) oder soll es erst deliberieren (natives Reasoning, mehr Deliberation vor dem Commit; höhere Compute-Kosten, langsamer, aber potenziell akkurater)?

Aktuelle Systeme lösen dieses Problem auf drei unbefriedigende Weisen:

1. **Fixe Budgets:** Jede Anfrage bekommt dasselbe Compute-Budget. Ineffizient, weil triviale Entscheidungen überkompensiert und schwierige unterkompensiert werden.

2. **Trainierte interne Mechanismen:** o1/o3-artige Modelle lernen intern, wann sie länger "nachdenken" sollen. Das funktioniert, ist aber (a) nicht interpretierbar, (b) nicht übertragbar zwischen Modellen, und (c) erfordert spezialisiertes Training.

3. **Prompt-Level-Allokation:** Arbeiten wie EAGer entscheiden *pro Prompt*, wie viel Compute investiert wird. In Agent-Settings, wo ein Prompt aus vielen sequentiellen Entscheidungsschritten besteht, ist diese Granularität zu grob — verschiedene Schritte innerhalb derselben Episode können radikal unterschiedliche Schwierigkeitsgrade haben.

### B. Kognitionspsychologische Grundlagen

Drei Theorietraditionen sind direkt relevant:

**Dual-Process Theory (Kahneman, 2011; Evans & Stanovich, 2013):**
Menschliches Denken operiert in zwei Modi — System 1 (schnell, automatisch, heuristisch) und System 2 (langsam, kontrolliert, deliberativ). Der Wechsel zwischen den Systemen wird durch *Überraschung*, *wahrgenommene Schwierigkeit* und *Fehlererwartung* ausgelöst.

**Expected Value of Control (EVC) Theory (Shenhav, Botvinick & Cohen, 2013):**
Das EVC-Framework formalisiert, wie das Gehirn entscheidet, *wie viel kognitive Kontrolle* in eine Aufgabe investiert wird. Die optimale Kontrollintensität maximiert:

    EVC(signal, intensity) = E[Reward(signal, intensity)] − Cost(intensity)

**Metacognition und Monitoring (Nelson & Narens, 1990; Flavell, 1979):**
Metacognition bezeichnet die Fähigkeit, das eigene Denken zu überwachen und zu steuern. Zentrale Signale sind Feeling of Knowing (FOK), Judgment of Learning (JOL) und Confidence Calibration. Diese Signale sind *informativ aber imperfekt* — Menschen nutzen heuristische Cues (Fluency, Familiarity, Coherence) als Proxy für tatsächliche Korrektheit. Entscheidend: Metacognitive Monitoring-Genauigkeit verschlechtert sich unter erhöhter kognitiver Last (Efklides, 2006) — eine Vorhersage, die direkt auf das agentische Setting übertragbar ist.

### C. Die Brücke: Metacognitive Signale als Compute-Allokator im Agent-Setting

Die zentrale theoretische Innovation besteht darin, metacognitive Signale als *Step-Level-Steuerungssignale für Compute-Tiefe* in sequentiellen LM-Agenten zu operationalisieren.

Der entscheidende Unterschied zu existierenden Ansätzen: Im agentischen Setting kommen Faktoren hinzu, die in Single-Turn-Settings nicht existieren:

1. **Kumulative Allokation:** Falsche Allokation bei Schritt 5 kann die Informationsbasis für Schritt 10 ruinieren.
2. **Memory-Abhängigkeit:** Die Qualität metacognitiver Signale kann sich über eine Episode verändern, weil der Kontext (History) wächst.
3. **Temporal Degradation:** Aus der Psychologie ist bekannt, dass Metacognitive Monitoring unter kognitiver Last an Genauigkeit verliert — das wachsende Context-Fenster im Agenten ist das computationale Äquivalent.

Die Analogie:
- **System 1** = Direkte Inferenz (1 Forward Pass)
- **System 2** = Deliberation (natives Reasoning / Chain-of-Thought)
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

**H2 (Adaptive Superiority Hypothesis):** Adaptive Compute-Allokation erreicht eine **Pareto-Verbesserung** gegenüber fixen Strategien: Bei **nicht-inferiorer Episode-Success-Rate** gegenüber der Always-Deliberate-Strategie (C2) erzielt sie **signifikant geringere Output-Tokens pro Episode** (und damit geringere Inferenzkosten), weil ein substantieller Anteil der Entscheidungsschritte trivial ist und korrekt von System-1-Inferenz gelöst werden kann.
*Psychologische Grundlage:* Pareto-Verteilung kognitiver Anforderungen in natürlichen Umgebungen (Anderson & Schooler, 1991).

**H3 (Temporal Degradation Hypothesis):** Die Kalibrierung metacognitiver Signale nimmt im Verlauf einer Episode ab. Bei späteren Entscheidungsschritten (mit längerem Context-Fenster) ist Token-Level-Entropie weniger prädiktiv für tatsächliche Korrektheit als bei frühen Schritten.
*Psychologische Grundlage:* Metacognitive Monitoring Accuracy sinkt unter erhöhter kognitiver Last (Efklides, 2006). Der wachsende Prompt-Context ist das LM-Äquivalent kognitiver Auslastung.

**H4 (Domain Modulation Hypothesis):** Metacognitive Signalqualität und/oder Allokationsmuster unterscheiden sich zwischen explorationslastiger sequentieller Navigation (TextWorld; partielle Beobachtbarkeit) und kombinatorischer Planung unter vollständiger Beobachtbarkeit (Tower of Hanoi), weil Fehlerquellen und Entropie-Signaturen domänenspezifisch sind.
*Psychologische Grundlage:* Domänenspezifität von FOK-Accuracy (Schwartz & Metcalfe, 2011); siehe `chapters/outline.md` für die aktuelle Domänenwahl.

---

## IV. Methodik

### A. Überblick: Vereinfachtes experimentelles Design

Die Studie implementiert ein kontrolliertes Experiment mit einem **2 × 3 × 2 Core-Design**:

- **Faktor 1 — Metacognitives Signal (2 Stufen):** Token-Level Entropy (TLE), Verbalisierte Konfidenz (VC)
- **Faktor 2 — Compute-Stufe (3 Stufen):** Direct Inference (C0), natives Reasoning via Thinking-Toggle (C1), **Self-Consistency Sampling** / **Majority Vote** (C2; Wang et al., 2022)
- **Faktor 3 — Task-Domäne (2 Stufen):** Text-Navigation (TextWorld), kombinatorische Planung (Tower of Hanoi)

**Vereinfachungen gegenüber v1 und Begründung:**

| v1 | v2 | Begründung |
|----|-----|-----------|
| 4 Compute-Stufen (Direct, Short CoT, Deep CoT, Tree Search) | 3 Stufen (Direct, Reasoning, **Self-Consistency Sampling**) | Statt prompt-gesteuerter CoT-Länge wird der native Thinking-Toggle des Modells genutzt (C1 = Reasoning an, C0 = Reasoning aus), was Gewichte, Tokenizer und Prompt-Oberfläche konstant hält; Tree Search ersetzt durch **Self-Consistency (Majority Vote)** (Wang et al., 2022): einfach zu implementieren, kein externes Scoring-Signal nötig; Snell et al. (2024) bleibt der Test-Time-Compute-Rahmenanker |
| 3 Signale (TLE, VC, SC) | 2 Signale Core (TLE, VC) | Semantic Consistency erfordert 5× Sampling pro Step — zu teuer für Core; bleibt Extension |
| 3 Domänen | 2 Domänen Core | Logical Reasoning bleibt Extension; 2 Domänen reichen für Cross-Domain-Test (RQ4) |
| 2 Modelle | 1 Modell Core | Zweites Modell bleibt Extension; 1 Modell reicht für Kernhypothesen |
| 30 Runs pro Bedingung | 5 Runs | Stufenspezifisch fixierte Decoding-Temperaturen (C0 0.3, C1 0.5, C2 0.7); TLE aus `raw_logprobs`; 5 Runs liefern stabile Mittelwerte |

Das Design hat zwei Phasen:

**Phase 1 — Calibration Mapping (RQ1, RQ3):** Für jede Kombination von Signal und Domäne wird die Kalibriertheit gemessen. Der Agent bearbeitet Aufgaben, wobei alle Compute-Stufen durchlaufen werden. Die Korrelation zwischen Signal und Korrektheit wird quantifiziert, aufgeschlüsselt nach Episode-Position (für RQ3).

**Phase 2 — Adaptive Allocation (RQ2, RQ4):** Die metacognitiven Signale steuern einen regelbasierten Allokator. Die Performance wird gegen Baselines verglichen.

### B. Prompt- und Strukturschema (XML-Tag-Strukturmarker)

Da die Studie *sequentielle Episoden* mit typischerweise 8–15 Schritten erzeugt, wächst der Prompt über eine Episode stark an (History aus Action+Observation-Paaren, ggf. Reset-Text, plus aktuelle Observation). Eine zentrale Operationalisierungsentscheidung ist daher, wie diese Komponenten im Prompt strukturiert werden. In Vorarbeiten wurde ein XML-Tag-basiertes Schema als primäres Strukturschema gewählt (statt rein visueller Trenner wie `=== HISTORY ===` oder reiner Markdown-Header), um die Prompt-Sektionierung auch bei langen Episoden stabil zu halten und Parsebarkeit für Mess- und Kontrollschritte zu erhöhen.

**Rationale.** Drei Gründe sind in diesem Setup methodisch relevant:

1. **Stabile Sektionszuordnung trotz wachsender History.** Ohne harte Marker müssen Modelle die Grenzen zwischen Reset-Observation, früheren Steps und aktueller Observation aus Fließtext rekonstruieren. Mit zunehmender Kontextlänge steigt das Risiko, dass relevante Information „in der Mitte verloren geht“ (*lost in the middle*; Liu et al., 2023) [VERIFY] oder dass der Agent die falsche Sektion als „current state“ behandelt. Explizite Tags reduzieren diese Ambiguität und verbessern die Replizierbarkeit des Orchestrierungs-Setups.

2. **Messrobustheit für Token-Level-Entropy (TLE).** TLE wird über die Tokens der *committeten* Step-Antwort (Action) berechnet. Ein klar markierter Übergang in eine `<action>`-Sektion reduziert die Wahrscheinlichkeit, dass vor der eigentlichen Action noch Meta-Text (z.B. „Looking at the current state…“) generiert wird, der den Messpunkt kontaminiert (zusätzliche Tokens, andere Logprob-Verteilungen).

3. **Robustheit der C1-Reasoning-Struktur.** In Compute-Stufe C1 erzeugt das Modell zunächst einen nativen Reasoning-Block (Thinking-Modus) und committet anschließend die Action. Tags erlauben, die committete Action sauber als eigene `<action>`-Sektion vom Reasoning-Block zu trennen, sodass der Reasoning-Trace nicht mit der Action oder der Episode-History kollidiert und der TLE-Messpunkt eindeutig bleibt.

**C1-spezifische Messintegritäts-Entscheidung (Reasoning/Action-Trennung).** Da C1 ein einzelner nativer Thinking-Durchlauf ist, wird TLE an der **committeden Action** gemessen, also am selben Extraktionspunkt wie in C0. Der vorausgehende Reasoning-Block ist nicht Teil des TLE-Messfensters und wird ausschließlich trace-intern geloggt. Damit bleibt die Messoberfläche zwischen C0 und C1 identisch, und es variiert allein die Menge des emittierten Reasonings (Toggle-Invarianz: Gewichte, Tokenizer und Prompt-Oberfläche bleiben konstant).

**Konkretes Schema (Referenzformat).** Der Prompt wird in klar getrennte Sektionen gegliedert:

```
<task_instructions>
… (task domain prefix, action constraints, output format) …
</task_instructions>

<episode_history>
  <reset_observation>…</reset_observation>
  <step index="1">
    <action>…</action>
    <observation>…</observation>
  </step>
  …
</episode_history>

<current_observation>
… (latest env observation) …
</current_observation>

<response_format>
Output exactly one imperative command. No reasoning, no explanation.
</response_format>
```

Zwei Designaspekte sind dabei absichtlich enthalten: (i) **Step-Index als Attribut**, um episodenpositionsabhängige Analysen (RQ3/H3) leichter an die Promptstruktur zu koppeln; (ii) **Verschachtelung** von `<step>` mit `<action>` und `<observation>`, um die Paarbeziehung explizit zu machen (reduziert Prompt-Drift bei späteren Steps).

**Wichtige Einschränkung.** Das Strukturschema wird *nicht* mit alternativen Markern gemischt. Ein halb-strukturiertes Prompting (z.B. XML plus zusätzliche `===`-Trenner) erhöht die Komplexität der Konventionen und kann die Modell-Compliance verschlechtern; daher wird das Schema in allen Compute-Stufen konsistent verwendet (C0/C1/C2 sowie VC-Elicitation).

### C. Operationalisierung der Metacognitiven Signale

**Signal 1 — Token-Level Entropy (TLE):**
Für den initialen Forward Pass (System-1-Response) wird die Entropie der Token-Verteilung über die Antwort-Tokens berechnet:

    H(t) = -Σ p(x_i) log p(x_i)

Aggregation über die gesamte Antwort via Mean und Max (Top-$K$ Renormalisierung, $K=20$; Phase-0-Sensitivität über $\{5,10,20\}$ vor Hauptdatenerhebung). TLE wird aus `raw_logprobs` auf der $T=1$-Skala berechnet; Decoding-Temperaturen sind stufenspezifisch (C0 0.3, C1 0.5, C2 0.7), die Entropie-Messung bleibt davon invariant. Hohe Entropie signalisiert Unsicherheit.

*Psychologische Analogie:* Processing Fluency als metacognitives Cue (Alter & Oppenheimer, 2009). Hohe Token-Entropie ist das computationale Äquivalent niedriger kognitiver Fluency.

*Abgrenzung zu EAGer:* EAGer nutzt Token-Entropie *innerhalb* eines Generierungsdurchlaufs, um zu entscheiden, wo neue Branches starten (Token-Level, Single-Turn). Wir nutzen die Entropie der *gesamten Step-Antwort* als aggregiertes Signal für die Compute-Stufe des *nächsten* Steps (Step-Level, sequentiell). Verschiedene Granularität, verschiedener Entscheidungsgegenstand.

**Signal 2 — Verbalisierte Konfidenz (VC):**
In einem separaten Follow-up-Aufruf nach Festlegung der Step-Aktion wird das Modell aufgefordert, die **Wahrscheinlichkeit einzuschätzen, dass die gewählte Aktion in dieser Situation korrekt ist** — nicht eine vage Bewertung von „Angemessenheit“ oder Optimalität. Der **judged context** ist die committete Aktionszeile, identisch über C0/C1/C2. Ausgabe ist **ein einzelner ganzzahliger Score von 0 bis 100**, mit **explizit verankerten Skalenenden** (0 = sicher falsch, 100 = sicher richtig); das entspricht der Probscore-/Likelihood-Elicitation und der empfohlenen Formulierung bei kleinen Sprachmodellen ohne Few-Shot-Zahlbeispiele (Yang, Tsai, & Yamada, 2024). Zur Erhöhung der Format-Compliance kann die Elicitation mit einer abschließenden **`Confidence:`**-Zeile erfolgen (Completion nach festem Marker; Yang et al., 2024; Lin et al., 2022). Bei Parse-Fehler: genau ein Reparse mit identischem Prompt bei Temperatur 0 (Erstversuch bei 0.2). Deliberatives natives Reasoning liegt **in der Compute-Stufe C1** (Thinking-Modus), nicht als zusätzlicher VC-only-Schritt, damit VC zwischen C0 und C1 vergleichbar zur jeweils committeden Action bleibt.

*Psychologische Analogie:* Feeling of Knowing (FOK) — ein explizites metacognitives Urteil (Koriat, 1993). FOK-Urteile sind informativ, aber systematisch verzerrt.

*Abgrenzung zu MeCo:* MeCo nutzt trainierte Probes auf Hidden-Layer-Aktivierungen (White-Box). Verbalisierte Konfidenz ist ein reines Output-Level-Signal (Gray-Box), das keinen Zugang zu Modell-Internals erfordert und somit modell-agnostisch einsetzbar ist.

### D. Compute-Stufen (Deliberationsintensität)

| Stufe | Beschreibung | Analogie | LM-Calls/Step (tertiär) |
|-------|-------------|----------|---------------|
| C0 — Direct | Einzelner Forward Pass, keine Elaboration | System 1 | 1 |
| C1 — Reasoning | Nativer Thinking-Durchlauf: Reasoning-Trace gefolgt von committeter Action (ein Call) | Deliberation | 1 |
| C2 — Self-Consistency | N=3 Reasoning-Generierungen + Majority Vote | Breadth / Sampling | N |

**Begründung der Vereinfachung:** Die drei Stufen sind aufsteigende Einstellungen *einer* Achse, der Menge deliberativen Reasonings vor dem Action-Commit (C0 = kein Reasoning, C1 = nativer Reasoning-Durchlauf, C2 = dasselbe Reasoning breitenskaliert). Primäres Compute-Maß sind Output-Tokens pro Episode; der LM-Call-Count dient nur als tertiäre Strukturtransparenz. C1 nutzt den nativen Thinking-Toggle desselben Modells, sodass sich C0 und C1 nur in der Reasoning-Menge unterscheiden (Gewichte, Tokenizer und Prompt-Oberfläche bleiben konstant). C2 ist **Self-Consistency Sampling** (Majority Vote; Wang et al., 2022): einfach zu implementieren, keine Branching-Logik nötig und **kein externes Scoring-Signal** erforderlich. Snell et al. (2024) bleibt als Test-Time-Compute-Rahmenanker für die generelle Idee compute-optimaler Inferenzskalierung.

**Mess-Symmetrie am Action-Punkt.** Über alle Stufen hinweg wird TLE an der committeden Action gemessen und dieselbe **Single-Line-Action-Ausgabeinstruktion** verwendet (einmalig definiert; single source of truth), sodass sich Unterschiede in TLE nicht aus unterschiedlichen Output-Format-Konventionen ergeben, sondern allein aus der vorausgehenden Reasoning-Menge.

### E. Adaptiver Allokator (Regelbasiert)

Der Allokator nutzt das metacognitive Signal s ∈ [0,1] (über ECDF auf dem Phase-1-Holdout normalisiert) und zwei Schwellenwerte θ₁ < θ₂:

    if s < θ₁:        → C0 (Direct — hohe Konfidenz)
    elif s < θ₂:      → C1 (Reasoning — moderate Unsicherheit)
    else:              → C2 (Self-Consistency — hohe Unsicherheit)

Schritt 0 ist fest auf C0 ohne vorheriges Signal; ab Schritt 1 steuert das Signal des vorherigen Steps die Allocation. Die Holdout-ECDF-Referenz wird nach der Schwellenwertsuche eingefroren und in Phase 2 identisch deployt.

Die Schwellenwerte werden einmalig auf fünf Holdout-Instanzen pro Domäne (im Task-Manifest markiert) per Grid-Search über Quantile 0.1–0.9 (θ₁ < θ₂; 36 Kandidatenpaare) optimiert. Zielgröße ist `step_level_proxy_v1`: pro Holdout-Step Stage-Matching gegen Phase-1-Zelloutcomes mit Fallback-Kaskade (exact → Mean über Runs → nearest position); Pareto-Front aus mittlerer Step-Korrektheit vs. summierten Step-Tokens, Tie-Break token-effizientester Punkt.

### F. Task-Domänen

**Domäne 1 — Text-Navigation (TextWorld):**
Standardisierte TextWorld-Instanzen mit kontrollierter Schwierigkeit. Der Agent muss ein Zielobjekt finden und eine Sequenz von Aktionen ausführen. Episoden haben typischerweise 8–15 Steps. Testet primär sequentielle Planung mit räumlichem Memory.

**Domäne 2 — Tower of Hanoi (Planung unter vollständiger Beobachtbarkeit):**
Klassisches kombinatorisches Puzzle mit vollständig sichtbarem Zustand; legale Züge sind klar definiert, aber optimalität ist nicht trivial. Ergänzt TextWorld durch eine Domäne mit anderer Fehlerstruktur (Planungs- vs. Explorationsfehler) bei gleichzeitig sequentieller Entscheidungsstruktur — konsistent mit RQ4/Cross-Domain in `chapters/outline.md`.

### G. Modellauswahl

**Core:** Qwen3-4B (oder funktional äquivalentes thinking-fähiges Modell zum Zeitpunkt der Durchführung). Das Modell verfügt über einen nativen Thinking-Toggle (`/think` an für C1/C2, `/no_think` für C0), der C0 und C1 am selben Modell ohne Architektur- oder Prompt-Oberflächen-Unterschiede vergleichbar macht (Yang et al., 2025).

**Extension:** Phi-3.5-mini-instruct (3.8B) als zweites Modell für Generalisierbarkeitstest.

Begründung für SLMs: (a) Lokale Inferenz ohne API-Abhängigkeit, (b) Reproduzierbarkeit, (c) Ressourcen-Constraints machen adaptive Allokation *relevanter* als bei großen Modellen, (d) Edge-Deployment-Implikationen.

### H. Abhängige Variablen

**Primär:**
- Task Success Rate (binär pro Episode)
- Output Tokens pro Episode (kumuliert; Summe über **alle** Modell-Outputs innerhalb der Episode)

**Sekundär:**
- Total Tokens Processed pro Episode (Input + Output; berücksichtigt wachsendes Episode-History-Contextfenster)
- Expected Calibration Error (ECE) der metacognitiven Signale
- Brier Score für Signalqualität
- Per-Step Allocation Distribution (wie oft wird C0/C1/C2 gewählt?)
- Signal Calibration by Episode Position (ECE aufgeschlüsselt nach Step-Index, für RQ3)

**Tertiär (struktureller Reproduzierbarkeitsindikator):**
- Anzahl Forward-Passes / LM-Calls pro Episode (z.B. C0=1 Call pro Step, C1=1, C2=N+Selector; dient der Strukturtransparenz, aber ist nicht das primäre Compute-Budget-Maß)

**Token-Accounting-Regeln (explizit; einmal definieren, dann überall verwenden):**
- **Primäre Compute-Achse:** Output-Tokens pro Episode (siehe oben). Der LM-Call-Count ist nur tertiärer Strukturindikator, nicht das Budget-Maß.
- **C1:** Sowohl die nativen Reasoning-Tokens (Thinking-Trace) als auch die Action-Tokens zählen zum Output-Compute. Es gibt **keinen separaten Verify-Output**; der Compute-Aufwand von C1 gegenüber C0 ist genau die zusätzliche Reasoning-Menge.
- **C2:** N Kandidaten-Outputs (jeweils Reasoning + Action) plus **Vote-Aggregation-Output** (Majority Vote) werden summiert.
- **VC-Elicitation (Phase 1):** zusätzlicher VC-Prompt-Output zählt zum Compute (struktureller Compute-Overhead von VC vs. TLE; als Konfound in §5.9 zu nennen).
- **Failure-Episoden:** Tokens werden bis zum Abbruch summiert; Episode-Success ist 0; Datenpunkt wird im Pareto-Reporting nicht ausgeschlossen.

**Report-Ebene (Episode vs. Step):**
- Episode-Level Output-Tokens: primäres Outcome für H2 (Pareto: Success vs. Output-Tokens).
- Step-Level Output-Tokens: deskriptiv für Allokationsanalyse (Compute-Verteilung über Steps).
- Kumulative Input-Tokens bis Step \(t\): relevant zur H3-Diskussion (wachsender Kontext als Load / Memory-Konfounder; als Limitation/Interpretationspunkt in §5.9 verankern).

### I. Baselines

| Baseline | Beschreibung | Funktion |
|----------|-------------|----------|
| Always-C0 | Immer Direct Inference | Lower Bound (Speed) |
| Always-C2 | Immer Self-Consistency (Majority Vote) | Upper Bound (Performance) |
| Random-Alloc | Zufällige Compute-Stufe pro Step | Kontrolle für den Effekt *informierter* Allokation |
| EAGer-Style | Episoden-Level-Allokation: Entropie des ersten Steps bestimmt ein festes Compute-Level für alle Steps der Episode | Kontrolle für den Mehrwert von *Step-Level*- vs. Prompt-Level-Allokation |

Die EAGer-Style-Baseline testet die zentrale Abgrenzung: Wenn Step-Level-Allokation besser abschneidet als Episoden-Level-Allokation, belegt das den Mehrwert der feineren Granularität — und damit den Beitrag dieser Arbeit gegenüber existierenden Ansätzen.

### J. Statistische Analysestrategie

**Für RQ1 (Kalibrierung):**
- AUROC (Diskrimination) und Brier Score nach Logit-Mapping für TLE (Kalibrierung) pro Domäne
- Reliability Diagrams und ECE (deskriptiv, nicht primär inferentiell)
- Vergleich TLE vs. VC via Cluster-Bootstrap auf ΔAUROC (H1a) und ΔBrier (H1b); Holm-Korrektur innerhalb der Hypothesenfamilien

**Für RQ2 (Adaptive Superiority):**
- **Primäres Reporting:** Pareto-Plot (Episode Success vs. Output Tokens pro Episode) mit Konfidenzintervallen auf beiden Achsen.
- **H2-Teststruktur:** kombinierter **Non-Inferiority-plus-Superiority-Test** relativ zu `Always-C2`:
  - **Non-Inferiority** auf Episode Success vs. `Always-C2` mit a-priori Marge \( \delta \) (in Kapitel 5 spezifiziert).
  - **Superiority** auf Output Tokens pro Episode (niedriger ist besser) vs. `Always-C2`.
- Paired Cluster-Bootstrap über Task-Instanzen; Mixed-Effects-Modelle als Fallback; **Task Difficulty als Kovariate**:
  - Tower of Hanoi: Disk-Anzahl.
  - TextWorld: Difficulty-Tier aus Manifest.
  - Alternativ/ergänzend: Per-Tier-Reporting der Pareto-Relation.

**Für RQ3 (Temporal Degradation):**
- Clustered logistic / GEE: $Y_{e,t} \sim z_{\mathrm{signal}} \times \mathrm{position\_norm} + \mathrm{position\_norm}$ mit $\mathrm{position\_norm} = t / \max(\mathrm{episode\_length} - 1, 1)$; roher `step_index` als Sensitivität
- Visualisierung: Kalibrierungs- und Diskriminationskurven nach Step-Position
- *Kostet null zusätzliche GPU-Stunden* — Daten fallen in Phase 1 ohnehin an

**Für RQ4 (Cross-Domain):**
- Difference-in-differences auf ΔAUROC (TLE − VC) zwischen Domänen; Cluster-Bootstrap
- Deskriptiver Vergleich der Allokator-Schwellenwerte zwischen Domänen

### K. Sample Size und Power

- 50 Task-Instanzen pro Domäne (2 Domänen = 100 Instanzen)
- 5 Wiederholungen pro Instanz pro Bedingung (stufenspezifische Decoding-Temperaturen: C0 0.3, C1 0.5, C2 0.7)
- 1 Modell im Core

Für den primären Vergleich (H2: Adaptive-TLE vs. Always-C2) mit erwarteter Effektstärke d = 0.5 ergibt eine Power-Analyse (α = .05, Power = .80) ~34 Instanzen pro Bedingung. Mit 50 Instanzen pro Domäne ist die Studie ausreichend gepowert.

---

## V. Priorisierungsstrategie (Core vs. Extension)

### Core (Muss für Abgabe vorhanden sein)

1. Signal-Kalibrierungsanalyse für TLE und VC auf Text-Navigation (TextWorld) und Tower of Hanoi mit 1 Modell
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

## VI-b. §5.7 Infrastructure — vLLM logprobs (preregistered)

**Pinned stack:** `vllm==0.19.1` on RunPod (V1 engine only). No version bump required for
`raw_logprobs`; the mode is an **engine argument** (`LLM(..., logprobs_mode="raw_logprobs")` or
`vllm serve --logprobs-mode raw_logprobs`), not a `SamplingParams` field.

**Semantics:** `raw_logprobs` = model output **before** logit processors (temperature, top_k, top_p,
penalties). Stage-specific decoding temperatures (C0 0.3, C1 0.5, C2 0.7) therefore do not confound TLE when `raw_logprobs` is active: TLE reads the pre-temperature distribution; sampling temperature affects which token is drawn, not the entropy scale (§5.2.1, §5.3).

**Validation:**

1. **Startup capability probe** (`VLLMWrapper`): fixed prompt, compare first-token TLE at T=0.3 vs
   T=1.0; hard-fail if `|dTLE| > 0.05` bits (engine not in raw mode).
2. **Pre-Phase-1 parity script** (`scripts/verify_backend_parity.py`): K-coverage + per-probe
   `|dTLE(T_low vs T_high)| ≤ eps`, with Same-T control and scaling-span diagnostics in JSON report.

**Preregistered tolerance:** `eps = max(0.05 bits, 3 × max_probe_same_T |dTLE|)`. Floor `0.05` bits
from RunPod control run (Qwen3-8B, RTX 5090, 2026-07-11); dynamic term absorbs fp16 request noise.

---

## VII. Vorläufige Gliederung

*Mental storyline (Spine):* LM agents make sequential decisions, but waste compute by treating every step equally. Cognitive psychology gives us a principled theory of when organisms invest effort — and a vocabulary (metacognition, EVC, fluency) that maps directly onto signals we can measure in LMs. Once we build that bridge, we can survey what's been tried, identify the gap — no one has tested these signals step-by-step in sequential agents — and fill it.

1. **Introduction**
   1.1 The Compute Allocation Problem in Language Model Agents
   1.2 Research Gap: From Single-Turn to Sequential Settings
   1.3 Structure of the Thesis

2. **Theoretical Background** *(Reihenfolge: Substrat → Theorie → Brücke)*
   2.1 Language Models as Sequential Decision Systems *(the substrate)*
       2.1.1 Transformer Architecture and Token-Level Inference
       2.1.2 Test-Time Compute: Definition, Budget Dimensions, and Scaling Mechanisms
       2.1.3 The Agent Loop: Action → Observation → Next Decision
       2.1.4 The Allocation Opportunity: Compute is Variable per Step, but Currently Fixed
   2.2 Cognitive Effort Allocation: A Framework from Human Cognition *(the theoretical lens)*
       2.2.1 Dual-Process Theory — System 1/2 as the Allocator Archetype
       2.2.2 Expected Value of Control — the Formal Cost–Benefit Model for "When to Think Harder"
       2.2.3 Metacognitive Monitoring and Control — FOK, JOL, Confidence; Fluency
       2.2.4 Temporal Degradation — Load Degrades Signal Accuracy; Unique to Sequential Settings
   2.3 Metacognition in Language Models *(the bridge)*
       2.3.1 Chain-of-Thought and Deliberative Reasoning as Computational System 2
       2.3.2 LM Metacognitive Capabilities and Their Limits
       2.3.3 Small Language Models: Capabilities and Constraints

3. **Related Work**
   3.1 Entropy-Based Compute Allocation (EAGer, Entropy Adaptive Decoding)
   3.2 Metacognitive Probes and Tool-Use Decisions (MeCo, AutoMeCo)
   3.3 Metacognitive Capabilities in LLMs (Li Ji-An et al., MGV Framework)
   3.4 Confidence Calibration in Language Models
   3.5 Agent Architectures with Variable Reasoning Depth
   3.6 Positioning of the Present Work

4. **Research Questions and Hypotheses**
   4.1 Derivation of Research Questions from Theory and Related Work
       4.1.1 RQ1: Comparative Calibration Quality of Entropy vs. Verbalized Confidence
       4.1.2 RQ2: Performance-Efficiency Effects of Step-Level Adaptive Allocation
       4.1.3 RQ3: Temporal Degradation of Signal Quality Across Episode Steps
       4.1.4 RQ4: Cross-Domain Stability (TextWorld vs. Tower of Hanoi)
   4.2 Hypotheses and Their Theoretical-Empirical Justification
       4.2.1 Signal Quality Hypotheses (H1–H2)
       4.2.2 Allocation and Temporal Dynamics Hypotheses (H3–H4)
   4.3 Factorial Design, Compute Stages, and Experimental Phases *(Kurzüberblick; Details in Kap. 5)*

5. **Methodology**
   5.1 Research Design Overview
   5.2 Operationalization of Metacognitive Signals
       5.2.1 Token-Level Entropy
       5.2.2 Verbalized Confidence
   5.3 Compute Stages and Deliberation Mechanisms
   5.4 Adaptive Allocation Mechanism
   5.5 Task Environments
       5.5.1 Text-Based Navigation
       5.5.2 Tower of Hanoi (Planning Under Full Observability)
   5.6 Baselines (incl. EAGer-Style)
   5.7 Model Selection and Infrastructure
   5.8 Statistical Analysis Plan
   5.9 Methodological Limitations

6. **Results: Signal Calibration Analysis (Phase 1)**
   6.1 Overall Calibration of Token-Level Entropy
   6.2 Overall Calibration of Verbalized Confidence
   6.3 Temporal Degradation Across Episode Steps
   6.4 Cross-Domain Comparison of Signal Quality

7. **Results: Adaptive Allocation Experiments (Phase 2)**
   7.1 Performance Comparison Against Baselines
   7.2 Compute Efficiency Analysis
   7.3 Step-Level vs. Prompt-Level Allocation (EAGer-Style Comparison)
   7.4 Allocation Patterns: When Does the Agent Choose to Deliberate?

8. **Discussion**
   8.1 Interpretation Through the Lens of Dual-Process Theory
   8.2 Metacognitive Signal Quality in Sequential vs. Single-Turn Settings
   8.3 Temporal Degradation: What Growing Context Does to Self-Monitoring
   8.4 Implications for Efficient Agent Design
   8.5 Limitations
   8.6 Ethical Considerations

9. **Conclusion and Future Work**
   9.1 Summary of Contributions
   9.2 Directions for Future Research
   9.3 Toward Metacognition-Aware AI Agents

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
- Setlur, A., Rajaraman, N., Levine, S., & Kumar, A. (2025). Scaling test-time compute without verification or RL is suboptimal. ICML 2025. arXiv:2502.12118.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., & Zhou, D. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. NeurIPS 2022.
- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv:1603.08983.
- Banino, A., et al. (2022). PonderNet: Learning to ponder. ICML 2022.
- Schuster, T., et al. (2022). Confident adaptive language modeling. NeurIPS 2022.
- Fu, Y., et al. (2025). DeepConf: Deep think with confidence. arXiv preprint.

### LM-Reasoning und Chain-of-Thought
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS 2022.
- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. NeurIPS 2022. arXiv:2205.11916.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate problem solving with large language models. NeurIPS 2023.
- Shinn, N., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. NeurIPS 2023.
- Madaan, A., et al. (2023). Self-Refine: Iterative refinement with self-feedback. NeurIPS 2023.
- OpenAI. (2024). Learning to reason with LLMs. https://openai.com/index/learning-to-reason-with-llms/
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948.

### Confidence Calibration in Language Models
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
- Yang, D., Tsai, Y.-H. H., & Yamada, M. (2024). On verbalized confidence scores for LLMs. arXiv:2412.14737.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? ICLR 2024.
- Guo, C., et al. (2017). On calibration of modern neural networks. ICML 2017.
- Lin, S., Hilton, J., & Evans, O. (2022). Teaching models to express their uncertainty in words. TMLR 2022.

### LM-Agenten und Sequentielle Entscheidungssysteme
- Yao, S., et al. (2023). ReAct: Synergizing reasoning and acting in language models. ICLR 2023.
- Zhou, A., et al. (2024). Language Agent Tree Search unifies reasoning, acting, and planning. ICML 2024.

### Small Language Models
- Abdin, M., et al. (2024). Phi-3 Technical Report. arXiv:2404.14219.
- Yang, A., et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
- Yang, A., et al. (2025). Qwen3 Technical Report. arXiv:2505.09388.

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
| Fehlende statistische Analysestrategie | Cluster-Bootstrap (AUROC, Brier, H2-Pareto), GEE/mixed logit (H3), Holm/BH-Multiplicity; ECE deskriptiv |
| Keine Power-Analyse | 50 Instanzen/Domäne, 5 Runs, Power >.80 für d=0.5 |
| "Regime Transition" vage definiert | Ersetzt durch "Temporal Degradation" (signal×`position_norm`-Interaktion; `position_norm` = $t/\max(\mathrm{episode\_length}-1,1)$) |
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