# #011 empty-chair. Deliberate owner-concealment in UK companies

The inverse model. We do not hunt the straw man the registry shows us; we hunt
the hole where an owner should be. Fabrication-of-absence: structures arranged
so that someone is deliberately NOT visible. The signal is the shape of the
disclosure itself: what is declared, what is exempted, what is routed through
corporate shells, what stays silent.

## AI-system card

| | |
|---|---|
| Prediction problem | Binary, PU-framed: is this company's beneficial-ownership disclosure deliberately evasive? |
| KPI | Investigator time: companies worth a human look surfaced per 100 reviewed (lift@k) |
| Proxy metric | PR-AUC + precision@k vs the blind rule; calibrated probabilities (brier) |
| Data sources | Companies House basic bulk, PSC bulk snapshot, CH API (case-control officers), ICIJ Offshore Leaks bulk, OpenSanctions |
| ML-system type | Batch scoring universe + on-demand scoring in the app (company number -> live CH API fetch -> score) |
| Consumed via | App: paste a UK company number, get score + evidence dossier (LLM-written, signal not verdict) |
| Monitoring | Inference logging to a dossier FG; drift on PSC-statement mix between snapshots |

## The label (and its honesty)

Positive = a company whose hidden interest was **later revealed**:
- UK-linked entities/officers matched into **ICIJ Offshore Leaks** (Panama,
  Paradise, Pandora, Bahamas: ~810k entities + relations, bulk CSV).
- UK companies linked to **OpenSanctions** persons where the designation
  post-dates formation (the owner was there all along, invisible).

Everything else is **unlabeled, not negative**. This is PU learning: hidden
owners that were never revealed sit in the "clean" class, so every measured
metric is a **lower bound** on true performance. Say this loud in the README.
The name match ICIJ<->Companies House is the weak joint: deterministic
normalization first (case, punctuation, LTD/LIMITED folding, address
canonicalization), an LLM entity-resolution pass only on the ambiguous band,
and a hand-audited sample of 100 matches to publish match precision.

## Class imbalance design (the plan)

Expected: thousands of positives vs 5.5M registered companies.

1. **Case-control sampling**: keep ALL positives; sample unlabeled controls
   stratified on incorporation-year x SIC-section x region, ~20:1. The
   stratification kills the cheap confounders (age, sector, geography) so the
   model must learn disclosure shape, not demographics.
2. **Class weights + calibration**: weighted loss, then isotonic/Platt
   calibration; decision threshold chosen for precision@k, never 0.5.
3. **Metrics that survive imbalance**: PR-AUC, precision@100 / @1000, lift over
   the blind rule. Accuracy and plain ROC-AUC never headline.
4. **Grouped split by formation-agent address cluster**: mills register
   thousands of near-identical companies; letting one mill straddle train/test
   is the leak that fakes the number (the paper_id scar from #010).
5. **Negative control**: shuffle-label run + a "demographics-only" model
   (year+SIC+region). If demographics-only scores close to the full model, the
   signal is population bias, not concealment shape. Publish both numbers.

## Blind baseline

The naive investigator rule: flag if `PSC statement = no-individual-PSC OR PSC
is a foreign corporate entity`. Headline = lift over that rule, same k.

## Features (at-observation, registry-only, the shape of the hole)

- **PSC shape**: statement type (no-PSC-exists, steps-not-completed, exemption),
  PSC is corporate vs natural, PSC chain depth, PSC residency/jurisdiction,
  time from incorporation to first PSC filing, PSC churn.
- **Officer shape** (case-control set via API): corporate officers, officer
  count, officers' directorship fan-out, officer nationality/residency mix,
  secretary-only patterns.
- **Formation shape**: agent-mill address (registered office shared by N
  thousand companies, computed from the bulk file), incorporation batch
  patterns, SIC codes (64209/70221/682xx holding clusters), name entropy.
- **Filing shape**: accounts category (dormant/micro/unaudited), overdue flags,
  confirmation-statement gaps.

One shared, pure extractor module (`chair_features.py`), imported by every
pipeline: no train/serve skew.

## Pipelines (FTI, connected only through the feature store)

### F. Feature pipelines
- **F1 `ingest-companies`** (blocked by: nothing). CH basic bulk zip (~5.5M) ->
  filter + stratified case-control frame -> `company_registry` FG. Raw zips in
  repo `data/` on HopsFS. Skills: hops-features, hops-fg.
- **F2 `ingest-psc`** (blocked by: F1). PSC bulk snapshot -> per-company PSC
  shape -> `psc_shape` FG. Skills: hops-features, hops-fg.
- **F3 `build-labels`** (blocked by: F1). ICIJ bulk + OpenSanctions -> GB
  matching -> `revealed_owner` FG (label, with match-confidence column).
  Skills: hops-features, hops-fg.
- **F4 `officer-fanout`** (blocked by: F1, F3; case-control set only). CH API
  (600 req/5min budget) -> `officer_network` FG. Skills: hops-features, hops-fg.

### T. Training pipeline
- **T `train-chair`** (blocked by: F1-F4). FV `empty_chair_fv` = registry JOIN
  psc_shape JOIN officer_network LEFT JOIN revealed_owner on company_number.
  EDA first (hops-eda): leakage sweep, demographics-only control. Model:
  HistGradientBoosting + calibrated logistic ensemble candidate; grouped split
  per the imbalance plan; eval JSON + ROC/PR/calibration/importance PNGs to the
  model registry as `empty_chair`. Skills: hops-fv, hops-train.

### I. Inference pipelines
- **I1 `score-universe`** (blocked by: T). Batch score the sampled universe ->
  `concealment_dossiers` FG with per-feature contributions. Skills:
  hops-batch-inference.
- **I2 app `emptychair`** (blocked by: I1). Server-rendered two-pane: paste a
  company number -> live CH API fetch -> shared extractor -> in-process score
  (small model, no KServe) -> evidence dossier, LLM-written from the numbers,
  signal not verdict. Inference logging on. Skills: hops-app,
  hops-online-inference (ODT pattern only).

## Order

F1 -> {F2, F3} -> F4 -> T -> I1 -> I2. Signal probe gate after T: if lift over
the blind rule is noise, publish the honest negative (the playbook owes one).

## Costs

Data: all free. LLM: entity-resolution band + dossier writing, est. < 10 USD.
