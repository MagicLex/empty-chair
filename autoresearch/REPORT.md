# autoresearch jul14: empty-chair PR-AUC round

Intent: maximize PR-AUC (average_precision_score) for `is_revealed` on the frozen
grouped holdout (GroupShuffleSplit by `office_group`, test_size 0.2, random_state 0).
Baseline: HGB + isotonic, faithful train_chair.py repro. 22 experiments, 10 min cap each.

## Leaderboard

| # | commit | change | PR-AUC | status |
|---|--------|--------|--------|--------|
| base | 9ae6d08 | HGB + isotonic (train_chair repro) | 0.286145 | keep |
| 1 | 2276fd1 | LightGBM, scale_pos_weight 19.7 | 0.314649 | keep |
| 2 | a1353d0 | drop calibration, fit on full train | 0.334363 | keep |
| 3 | c4e8a89 | interaction features between tells | 0.340631 | keep |
| 4 | dbfdb63 | lr 0.02, 1500 trees, 127 leaves | 0.339675 | discard |
| 5 | 966f013 | lr 0.03, 1000 trees | 0.343236 | keep |
| 6 | 01e7913 | unweighted (scale_pos_weight 1) | 0.367005 | keep |
| 7 | 177d01d | LGBM native categoricals | 0.355067 | discard |
| 8 | 486ef7c | full sic_code as ordinal categorical | 0.365268 | discard |
| 9 | 73c62ee | deeper unweighted (127 leaves, 2000 trees) | 0.337088 | discard |
| 10 | ad566d1 | shallower regularized (31 leaves, mcs 80, l2 5) | 0.368428 | keep |
| 11 | ce2ab5c | heavier regularization | 0.367609 | discard |
| 12 | d586eb4 | XGBoost hist depth 6 | 0.354958 | discard |
| 13 | cc94db0 | 5-seed LGBM soft-vote ensemble | 0.370600 | keep |
| 14 | 500d8d9 | prune mill/office_count/dormant (bias audit) | 0.368507 | discard |
| 15 | 24b86ed | 3xLGBM + 2xHGB heterogeneous vote | 0.369122 | discard |
| 16 | 429b4a4 | monotone-up constraints on tells | 0.370057 | discard |
| 17 | 6e9f72e | grouped OOF TE: post_area, sic_section | 0.370872 | keep |
| 18 | 3b059df | TE: add full sic_code | 0.374742 | keep |
| 19 | 60a85a8 | TE smoothing 20 -> 10 | 0.373970 | discard |
| 20 | 4b8e66e | TE on post_area x sic_section cross | 0.372631 | discard |
| 21 | f5a0540 | 10-seed ensemble | 0.376026 | keep |
| 22 | a3a9d7c | lr 0.02, 1500 trees in 10-seed ensemble | 0.376826 | keep |

Keeps: 11. Discards: 11. Crashes: 0.

## Winner

Commit a3a9d7c (registered as `autoresearch_jul14` v19): 10-seed LightGBM soft-vote,
lr 0.02, 1500 trees, 31 leaves, min_child_samples 80, l2 5, unweighted, no calibration,
tell-interaction features, grouped OOF target encoding of post_area / sic_section / sic_code.

**PR-AUC 0.376826 vs baseline 0.286145 (+31.7%). Holdout base rate 0.0471 (8x lift).**

## Honesty controls: PASS

- Shuffle-label: 0.047053, collapses to base rate 0.0471 exactly. No leakage path
  through the pipeline, TE included.
- Demographics only (incorporation year, SIC, region, same TE treatment): 0.080799,
  21% of the winner, far under the 30% taint line. The concealment shape carries
  the signal, not who/where/when the company is.

## Interpretation

Three moves carried almost everything: dropping scale_pos_weight (+0.024, class
weighting distorts leaf values and buys nothing for a pure ranking metric), dropping
isotonic calibration to reclaim the 25% calibration slice for fitting (+0.020, a
monotone map cannot improve PR-AUC by construction, the baseline was paying data for
nothing), and switching HGB to LightGBM (+0.028). Everything after 0.368 is grinding
inside a roughly +/-0.002 noise floor of a holdout with ~1000 positives; the seed
ensemble and target encoding gains are real but small, and single-run deltas at that
scale should not be over-read.

The bias-audit experiment (exp 14) is the most useful negative result: removing
is_mill_address, office_company_count and accounts_dormant costs only 0.002 PR-AUC.
The model does not ride the formation-mill confound; PSC concealment shape
(psc_corporate_only, psc_silence, psc_absent and friends) does the work, which
matches the demographics control.

Caveat that bounds all of this: the label is PU-learned via name matching, so
0.377 is precision against later-revealed owners only. Unrevealed true positives
sitting in the negatives depress every number here uniformly; the ranking between
experiments is trustworthy, the absolute PR-AUC is a lower bound.

## Postscript (2026-07-14, same day)

The winner was retired hours later. Scoring the full universe showed its top 1%
was 99.3% real estate against 13.8% of training positives; the raw sic_section
categorical carried the sector, and no experiment in this round ablated it (the
TE ablations left it in CAT). The sector-blind rebuild is `empty_chair` v11,
PR-AUC 0.199. Trail: docs/bias-audit.md and the README model cards.
