# empty-chair bias / confound audit

Pre-publication check. No protected attribute is inferred. The question: does the score track concealment, or a legitimate corporate/property structure that correlates with both the labels and certain business communities?

## 1. SIC concentration in the top 1%

| SIC | activity | share of top 1% | lift vs population |
|---|---|--:|--:|
| 68 | real estate | 34.7% | 3.2x |
| 47 | retail | 10.9% | 1.4x |
| 99 | dormant | 10.1% | 5.2x |
| 62 | IT | 7.3% | 1.4x |
| 98 | other | 5.2% | 2.3x |
| 46 | other | 3.8% | 1.1x |
| 41 | construction | 2.5% | 0.6x |
| 70 | head office / management consulting | 2.2% | 0.4x |
| 56 | other | 2.1% | 0.5x |
| 58 | other | 1.5% | 1.8x |

## 2. Structural composition: top 1% vs whole population

| trait | top 1% | population | positives (labels) |
|---|--:|--:|--:|
| psc_absent | 5.3% | 6.2% | 7.9% |
| psc_silence | 7.0% | 6.5% | 7.6% |
| psc_corporate_only | 3.0% | 5.9% | 17.2% |
| psc_foreign_corporate | 0.9% | 1.0% | 5.3% |
| is_mill_address | 45.4% | 28.1% | 29.0% |
| is_holding_sic | 12.8% | 12.1% | 25.1% |
| accounts_dormant | 54.3% | 37.2% | 36.7% |
| active | 99.5% | 98.0% | 97.6% |

## 3. Confound isolation (mean score)

- holding-SIC companies: 0.112  vs  non-holding: 0.122
- foreign-corporate PSC: 0.113  vs  none: 0.121

## Reading

If the top 1% is dominated by real-estate / holding SIC at high lift, and holding-SIC and foreign-corporate alone move the mean score a lot, then the model is substantially a **corporate-structure detector**: it finds SPV / holding / multinational vehicles, which include both genuine concealment and ordinary legitimate structuring (property developers, REITs, group treasuries). It cannot tell intent apart. Publishing named individuals as concealment-linked on this basis is not defensible; the honest use is anonymous, structural triage.