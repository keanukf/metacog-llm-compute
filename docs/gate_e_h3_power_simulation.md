# Gate E (WEICH) — H3-Power-Simulation

**Datum:** 2026-07-17
**Branch/Commit:** `feat/gate-d-calibration`
**Skript:** [`scripts/h3_power_simulation.py`](../scripts/h3_power_simulation.py)
**Seed-Quelle (Pilot-ICC/Entropie):** `data/results/instrument_validation/phase1_20260714_105004/` (72 reale Episoden, dieselbe Gate-C-Pilotquelle wie `docs/gate_e_rehearsal.md`)
**Report-Artefakt (lokal, `data/results/` ist gitignored):** `data/results/gate_e_h3_power/h3_power_simulation.json`

**Zweck:** `blueprints/gate_p1_readiness.md`, Gate E, WEICH-Punkt „H3-Power-Simulation": §5.8 sieht
eine simulationsbasierte Power-Prüfung für die H3-Interaktion vor, geseedet mit Pilot-ICC und
Entropieverteilung, weil der konfirmatorische H3-Test (geclustertes Logit-/GEE-Modell,
Instanz-Clustering) keine geschlossene Power-Formel besitzt und der Pilot selbst (9–12 Cluster je
Domäne) die Interaktion nicht mit brauchbarer Präzision schätzen kann (§5.9 nennt das explizit,
unter Verweis auf X. Zhao 2026 zu unterpowerten stufenindizierten Interaktionstests). Dieser
Durchlauf **führt** die Simulation durch (statt die Limitation nur zu benennen) — der WEICH-Punkt
ist damit erledigt im Sinne der Gate-E-Formulierung „durchführen **oder** die Limitation aktiv
wählen".

---

## 1. Getestetes Modell (exakt aus §5.8 / Code, nicht neu erfunden)

H3 (`chapters/05_methodology.md` §5.8, „H3: temporal degradation"): geclustertes Logit-Modell von
$Y_{e,t}$ auf das standardisierte Signal $z$, `position_norm = t / max(episode_length-1, 1)`, und
deren Produkt, geschätzt pro Signal und Domäne mit Instanz-Level-Clustering (GEE, Exchangeable
Working-Korrelation, `Binomial`-Familie). Degradation entspricht einem **negativen**
Interaktionskoeffizienten. Konfirmatorisch in TextWorld, exploratorisch in Tower of Hanoi. H3
bildet mit H1b eine eigene kleine Holm-Familie; da H3 selbst pro Signal geschätzt wird (TLE, VC),
enthält seine eigene Familie **zwei** Tests — unter Holm braucht der strengere der beiden
$\alpha/2 = .025$ (einseitig), der lockere (bedingt auf Ablehnung des ersten) $\alpha = .05$.

Die Simulation ruft **exakt** die Produktionsfunktion `src/analysis/inference.py::fit_h3_model`
auf jedem simulierten Datensatz auf — kein Nachbau des Modells, kein separater
Analyse-Code-Pfad. Das ist bewusst so gewählt: die Power-Zahl gilt damit für den Code, der auch
die echte konfirmatorische Analyse fahren wird, nicht für eine Idealisierung davon.

**Bekannte Werktreue-Einschränkung (nicht behoben, da außerhalb des Auftrags):** `fit_h3_model`
zentriert das Signal (`z_c = z - mean(z)`), skaliert aber nicht durch die Standardabweichung — es
ist "centered", nicht im strengen Sinn "standardised" trotz der §5.8-Formulierung. Die Simulation
umgeht das sauber, indem sie dem Modell bereits standardisierte (SD=1) synthetische Signalwerte
füttert; nach der Zentrierung (Mittelwert bei N~750 Episoden/Domäne praktisch 0) entspricht der
geschätzte Koeffizient dann direkt einem Pro-SD-Effekt. Für die echten Phase-1-Daten bleibt diese
Diskrepanz zwischen Modellcode und Kapiteltext bestehen und ist hier nur dokumentiert, nicht
gefixt (kein Analyse-Code-Bugfix ohne expliziten Auftrag).

---

## 2. Pilot-seeded Statistiken (aus den echten 72 Episoden)

| Domäne | n Steps | n Cluster | Basisrate $Y=1$ | ICC (GEE, `dep_params`) | ICC (ANOVA-Kreuzcheck) | TLE Mean/SD | VC Mean/SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| TextWorld | 703 | 12 | 0.115 | **0.199** | 0.223 | 0.0328 / 0.1074 | 74.1 / 20.8 |
| Tower of Hanoi | 659 | 12 | 0.270 | **0.014** | 0.021 | 0.0398 / 0.0880 | 58.5 / 31.7 |

**ICC-Methode:** primär ein interzept-only GEE (Exchangeable, Binomial) auf `y_optimal`, geclustert
über `instance_key` — `dep_params` **ist** exakt der Clustering-Parameter, den die konfirmatorische
GEE-Analyse selbst schätzen würde (methodisch am saubersten angebunden, nicht nur ein generischer
ICC-Schätzer). Kreuzcheck über die klassische ANOVA-basierte ICC(1) auf der 0/1-Outcome-Skala
(Standardkonvention für binäre Cluster-RCT-Outcomes) — beide Methoden liegen nah beieinander in
beiden Domänen, was Vertrauen in die ICC-Schätzung gibt.

**Auffälligkeit:** TextWorld hat eine ~14× höhere Cluster-Korrelation als ToH (0.199 vs. 0.014).
Plausibel: ToH-Episoden nähern sich stärker einem instanz-unabhängigen Erfolgs/Fehler-Muster
(planungsgetrieben, weniger stabile pro-Instanz „Charakteristik" als TextWorlds
explorationsgetriebene, pro-Spielinstanz variierende Schwierigkeit). Diese ICC-Differenz treibt
einen großen Teil des Power-Unterschieds zwischen den Domänen unten.

**Reale (verrauschte) Pilot-H3-Punktschätzer** (`fit_h3_model` auf den 72 echten Episoden, **nur
Kontext, nicht als wahrer Effekt verwendet** — n=12 Cluster/Domäne kann die Interaktion nicht
verlässlich schätzen, genau das Problem, das diese Simulation umgeht):

| Domäne | Signal | Interaktion (roh) | Interaktion (SD-standardisiert) | p (zweiseitig) | Richtung |
|---|---|---:|---:|---:|---|
| TextWorld | TLE | −8.459 | **−0.909** | 0.0068 | konsistent mit Degradation |
| Tower of Hanoi | TLE | +31.831 | **+2.801** | 0.157 | **gegen** die Degradationsrichtung, nicht signifikant |

Der TextWorld-Punktschätzer zeigt (bei aller Vorsicht wegen n=12) eine große, in die erwartete
Richtung weisende Interaktion; der ToH-Punktschätzer zeigt das Gegenteil, nicht signifikant — beide
Zahlen sind mit so wenigen Clustern kaum interpretierbar, sie dienen hier nur als grober
Größenordnungs-Anker für das Simulationsraster unten.

---

## 3. Simulationsdesign und explizite Annahmen

**Geplantes Phase-1-Design** (`configs/experiment_core.yaml`): 50 Instanzen/Domäne, 5
Runs/Bedingung × 3 Compute-Stages (C0/C1/C2) → **15 Episoden/Instanz, 750 Episoden/Domäne**,
gepoolt über die Stages (so wie `fit_h3_model` selbst nicht nach `compute_stage` filtert).

**Generatives Modell** (Random-Intercept-Logit, pro Domäne × Signal):

$$\text{logit}(P(Y_{e,t}=1)) = \beta_0 + \beta_z \cdot z_t + \beta_{pos} \cdot \text{position\_norm}_t + \beta_{int} \cdot z_t \cdot \text{position\_norm}_t + b_i$$

mit $b_i \sim \mathcal{N}(0, \sigma_b^2)$ (Instanz-Zufallseffekt), $z_t \sim \mathcal{N}(0,1)$ iid
pro Step (**vereinfachende Annahme:** kein Within-Episode-Autokorrelationsterm auf dem Signal
selbst — nur die Instanz-Ebene ist korreliert; die reale TLE-Serie innerhalb einer Episode dürfte
etwas autokorreliert sein, was hier nicht modelliert wird und die Power tendenziell leicht
**überschätzen** könnte, da echte Daten zusätzliche redundante Korrelation zwischen benachbarten
Steps hätten, die die effektive Stichprobengröße weiter reduziert).

| Parameter | TextWorld | Tower of Hanoi | Herkunft |
|---|---|---|---|
| $\sigma_b^2$ | 0.817 | 0.045 | aus ICC (GEE `dep_params`) über die logistisch-normale Latent-ICC-Konvention $\sigma_b^2 = \frac{\text{ICC}}{1-\text{ICC}}\cdot\frac{\pi^2}{3}$ |
| $\beta_z$ (TLE) | 0.30 | 1.20 | Pilot-GEE-Punktschätzer (standardisiert) + Cohen's-d-Kreuzcheck aus `preanalysis_screen.json`, gerundet |
| $\beta_z$ (VC) | 0.25 | 1.00 | dito, VC-Pilotwerte |
| $\beta_{pos}$ | −0.6 | −0.9 | Pilot-`p_c`-Koeffizienten (grober Anker, Nuisance-Parameter, nicht das H3-Ziel) |
| Zielbasisrate | 0.115 | 0.270 | Pilot `y_optimal`-Rate, per numerischer Intercept-Kalibrierung getroffen |
| Episodenlänge | uniform 8–15 Steps | Bootstrap aus realen Pilot-Längen (Ø ≈18, viele nahe dem 20-Step-Cap) | **siehe Warnung unten** |

**Wichtigste Annahme, explizit markiert — Episodenlänge:** Für TextWorld wird der **Gate-D-Zielkorridor**
(8–15 Steps, `blueprints/gate_p1_readiness.md`, load-bearing für H3-Positionsauflösung) verwendet,
**nicht** die rohe Pilot-Verteilung — letztere liegt nahe am 20-Step-Cap, weil die Pilot-Daten vor
der TextWorld-Schwierigkeitskalibrierung (Gate D, noch offen) entstanden sind. Für ToH existiert
**kein** expliziter Längenkorridor (nur ein 30–50-%-Erfolgsziel), daher wird hier auf die
Bootstrap-Resamples der realen Pilot-ToH-Längen zurückgegriffen — mit der Einschränkung, dass auch
ToHs Disk-Zahl/Partial-Start-Kalibrierung (Gate D) noch nicht eingefroren ist und die kalibrierten
Episoden am Ende **kürzer** ausfallen könnten. **Das bedeutet: die ToH-Zahlen unten sind mit realen,
eher zu langen (Positionsauflösung begünstigenden) Episoden simuliert — sie sind ein optimistischer
Kontextwert für die exploratorische ToH-Seite, nicht die belastbarste Aussage dieses Reports.** Die
TextWorld-Zahlen (Gate-D-Zielkorridor, konfirmatorische Domäne) sind der primäre, verlässlichere
Befund.

**Kalibrierung des Intercepts:** $\beta_0$ wird pro Zelle numerisch (Bisektion, 40 Iterationen) so
gesetzt, dass der analytische Mittelwert von $\text{expit}(\text{logit})$ über eine
Probe-Population die Pilot-Basisrate trifft; für alle $\beta_{int}$ derselben Zelle wiederverwendet
(der Effekt von $\beta_{int}$ auf die marginale Rate ist klein — dokumentierte Vereinfachung).

**Multiplizität/Alpha:** einseitig $\alpha=.05$ nominal (§5.8-Konvention: obere/untere Grenze eines
90-%-Intervalls) sowie der Holm-konservative Wert $\alpha=.025$ für die strengere Position in
H3s Zweier-Familie (TLE + VC). Beide werden berichtet.

**Fitting-Engine:** echte `statsmodels`-GEE-Fits (Exchangeable, Binomial) auf jedem simulierten
Datensatz — keine Bootstrap-Approximation, keine geschlossene Formel. Multiprocessing über 7
Worker-Prozesse (BLAS-Thread-Oversubscription per `OMP_NUM_THREADS=1` etc. unterbunden).

**Validität der Simulationsmaschinerie (Typ-I-Fehler-Check):** bei $\beta_{int}=0$ (wahre
Nullhypothese) liegt die empirische Ablehnrate bei $\alpha=.05$ zwischen 0.024–0.048 und bei
$\alpha=.025$ zwischen 0.004–0.025 über alle vier Domäne×Signal-Zellen — nahe am, tendenziell
leicht **konservativ unter** dem nominalen Niveau. Das spricht gegen eine anti-konservative
(zu optimistische) Verzerrung der berichteten Power-Zahlen.

---

## 4. Ergebnisse: Power-Kurve (primär: TLE, volles Raster, 400 Replikate/Zelle)

| $\beta_{int}$ (wahr) | TextWorld Power (α=.05) | TextWorld Power (α=.025) | ToH Power (α=.05) | ToH Power (α=.025) |
|---:|---:|---:|---:|---:|
| 0.00 | 0.033 | 0.013 | 0.048 | 0.025 |
| −0.05 | 0.123 | 0.068 | 0.155 | 0.090 |
| −0.10 | 0.215 | 0.135 | 0.358 | 0.258 |
| −0.15 | 0.348 | 0.228 | 0.510 | 0.418 |
| −0.20 | 0.523 | 0.398 | 0.780 | 0.673 |
| −0.30 | 0.790 | 0.688 | 0.975 | 0.945 |
| −0.40 | 0.930 | 0.873 | 1.000 | 1.000 |
| −0.50 | 0.990 | 0.980 | 1.000 | 1.000 |
| −0.75 | 1.000 | 1.000 | 1.000 | 1.000 |
| −1.00 | 1.000 | 1.000 | 1.000 | 1.000 |

**80-%-Power-Schwelle (interpoliert):**

| Domäne/Signal | $\lvert\beta_{int}\rvert$ bei 80 % Power (α=.05) | bei 80 % Power (α=.025, Holm) | Als Anteil des Haupteffekts $\lvert\beta_z\rvert$ |
|---|---:|---:|---:|
| TextWorld / TLE | **0.307** | 0.361 | **≈102–120 %** von $\beta_z=0.30$ |
| ToH / TLE | 0.210 | 0.247 | ≈18–21 % von $\beta_z=1.20$ |
| TextWorld / VC | 0.286 | 0.334 | ≈114–134 % von $\beta_z=0.25$ |
| ToH / VC | 0.225 | 0.243 | ≈23–24 % von $\beta_z=1.00$ |

**Sekundärraster (VC, 250 Replikate/Zelle):**

| $\beta_{int}$ | TextWorld VC (α=.05) | ToH VC (α=.05) |
|---:|---:|---:|
| 0.00 | 0.024 | 0.032 |
| −0.15 | 0.328 | 0.600 |
| −0.30 | 0.848 | 1.000 |
| −0.50 | 0.996 | 1.000 |
| −1.00 | 1.000 | 1.000 |

---

## 5. Interpretation

**Der zentrale, konfirmatorische Fall ist TextWorld/TLE.** Bei $\beta_z=0.30$ (der pilot-verankerte
Haupteffekt) bedeutet eine Interaktion von $\beta_{int}=-0.307$, dass das Signal am **Episodenende**
($\text{position\_norm}=1$) praktisch **keine** Restbeziehung zur Korrektheit mehr hat
($\beta_z+\beta_{int}\approx 0$) — vollständige Degradation. **Genau dieser Fall — vollständiger
Signalverlust bis Episodenende — ist der Punkt, an dem das Design 80 % Power erreicht.** Für
plausiblere, moderate Degradationsgrade — z. B. eine Halbierung der Signal-Korrektheits-Steigung
über die Episode ($\beta_{int}\approx-0.15$, 50 % von $\beta_z$) — liegt die Power bei nur **≈35 %**
(α=.05) bzw. **≈23 %** (α=.025, Holm-konservativ). Ein Viertel-Effekt ($\beta_{int}\approx-0.075$,
zwischen den simulierten Punkten −0.05 und −0.10, Power ≈17–22 %) wäre nahezu unentdeckbar.

**Tower of Hanoi zeigt scheinbar bessere Power, aber mit einer wichtigen Einschränkung.** Die
niedrigere Cluster-ICC (0.014 vs. 0.199) und der größere Haupteffekt ($\beta_z=1.2$) treiben die
80-%-Schwelle auf einen kleineren Absolutwert und einen kleineren Anteil des Haupteffekts (~20 %
statt ~110 %). Das widerspricht auf den ersten Blick der eigenen These-Begründung, warum ToH nur
exploratorisch ist („kurze Episoden, wenig Positionsauflösung", §5.8) — der Widerspruch löst sich
aber durch die oben genannte Episodenlängen-Annahme auf: die Simulation nutzt die **realen, noch
nicht schwierigkeitskalibrierten** ToH-Pilotlängen (Ø≈18, nahe dem 20-Step-Cap), die länger sind
als TextWorlds hier verwendeter **kalibrierter Zielkorridor** (8–15). Sobald Gate D die
ToH-Diskzahl/Partial-Start final auf den 30–50-%-Erfolgskorridor kalibriert, dürften echte
ToH-Episoden kürzer und die Positionsauflösung entsprechend geringer ausfallen als hier simuliert —
die ToH-Zahlen sind also ein **oberer, optimistischer** Kontextwert, nicht die belastbarste
Grundlage für eine Entscheidung. Da ToH ohnehin nur exploratorisch getestet wird (keine
Holm-Familie, BH auf Ebene der gesamten exploratorischen Schicht), ist dieser Punkt für die
Gate-Entscheidung nachrangig gegenüber dem TextWorld-Befund.

**VC verhält sich ähnlich wie TLE** (leicht niedrigerer Haupteffekt-Anker, daher tendenziell etwas
niedrigere absolute Power-Schwellen in TextWorld, aber dieselbe qualitative Aussage: nur große
Interaktionseffekte werden zuverlässig entdeckt).

---

## 6. Empfehlung (Befund, keine Entscheidung — die trifft der Nutzer)

Das aktuell geplante Design (50 Instanzen/Domäne, 5 Runs/Bedingung, 3 Stages gepoolt) ist für den
**konfirmatorischen** H3-Test in TextWorld **nur für sehr große Interaktionseffekte** (Größenordnung
des vollen Haupteffekts, d. h. praktisch vollständiger Signalverlust bis Episodenende) mit ≈80 %
Power ausgestattet. Für plausiblere partielle Degradationsgrade (25–50 % Abschwächung der
Signal-Steigung) liegt die Power nur bei etwa 15–35 % — ein reales Risiko, dass ein tatsächlich
vorhandener, aber moderater Degradationseffekt im konfirmatorischen Test **nicht** signifikant wird,
selbst wenn die Alternativhypothese inhaltlich zutrifft. Dies ist konsistent mit §5.9s bereits
formulierter Warnung („die Interaktionstests von H3 und H4 verlangen mehr Power als Haupteffekte")
und mit dem in Kapitel 4 zitierten Befund von X. Zhao (2026) zu unterpowerten Interaktionstests in
verwandten Settings — die Simulation liefert jetzt eine **quantifizierte** Version dieser bereits
qualitativ benannten Sorge.

Diese Zahlen sind kein Grund, das Design vor Phase 1 zu ändern (N zu erhöhen ist ohnehin durch das
GPU-Budget begrenzt, siehe Gate F), aber eine belastbare Grundlage für die Interpretation eines
nicht-signifikanten H3-Ergebnisses in Kapitel 6/7: ein Nullbefund bei H3 sollte **nicht** automatisch
als „keine Degradation" gelesen werden, sondern als „keine *große* Degradation zuverlässig
ausgeschlossen, moderate Degradation bleibt plausibel und unterpowert" — eine Formulierung, die in
§5.9 (Limitations) ergänzt werden könnte, falls gewünscht (nicht in diesem Durchlauf vorgenommen,
da Prosaänderungen im Thesis-Repo nicht Teil dieses Auftrags waren).

---

## 7. Reproduzieren

```bash
python3 scripts/h3_power_simulation.py \
  --run-dir data/results/instrument_validation/phase1_20260714_105004 \
  --out data/results/gate_e_h3_power/h3_power_simulation.json \
  --n-reps 400 --n-reps-secondary 250 --workers 7
```

Laufzeit: ~2 Minuten auf einem 8-Core-Laptop (echte `statsmodels`-GEE-Fits, aber kleine
Datensätze und starke Parallelisierung über Replikate). Volles JSON-Artefakt (Pilot-Stats,
Design-Zellen mit Begründung, komplettes Power-Raster pro Zelle, 80-%-Kreuzungspunkte) liegt lokal
unter `data/results/gate_e_h3_power/h3_power_simulation.json` (gitignored, wie alle
`data/results/`-Artefakte — dieses Dokument ist die archivierte Evidenz).
