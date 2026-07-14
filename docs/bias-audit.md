# empty-chair bias / confound audit

Pre-publication check. No protected attribute is inferred. The question: does the score track concealment, or a legitimate corporate/property structure that correlates with both the labels and certain business communities?

## 1. SIC concentration in the top 1%

| SIC | activity | share of top 1% | lift vs population |
|---|---|--:|--:|
| 68 | real estate | 97.5% | 8.9x |
| 98 | other | 2.3% | 1.0x |
| 47 | retail | 0.2% | 0.0x |
| 43 | other | 0.0% | 0.0x |
| 45 | other | 0.0% | 0.0x |
| 99 | dormant | 0.0% | 0.0x |
| 64 | financial holding / trusts | 0.0% | 0.0x |
| 96 | other | 0.0% | 0.0x |

## 2. Structural composition: top 1% vs whole population

| trait | top 1% | population | positives (labels) |
|---|--:|--:|--:|
| psc_absent | 0.5% | 6.2% | 7.9% |
| psc_silence | 2.1% | 6.5% | 7.6% |
| psc_corporate_only | 1.7% | 5.9% | 17.2% |
| psc_foreign_corporate | 0.5% | 1.0% | 5.3% |
| is_mill_address | 38.0% | 28.1% | 29.0% |
| is_holding_sic | 46.9% | 12.1% | 25.1% |
| accounts_dormant | 63.4% | 37.2% | 36.7% |
| active | 99.7% | 98.0% | 97.6% |

## 3. Confound isolation (mean score)

- holding-SIC companies: 0.337  vs  non-holding: 0.211
- foreign-corporate PSC: 0.185  vs  none: 0.226

## Reading

If the top 1% is dominated by real-estate / holding SIC at high lift, and holding-SIC and foreign-corporate alone move the mean score a lot, then the model is substantially a **corporate-structure detector**: it finds SPV / holding / multinational vehicles, which include both genuine concealment and ordinary legitimate structuring (property developers, REITs, group treasuries). It cannot tell intent apart. Publishing named individuals as concealment-linked on this basis is not defensible; the honest use is anonymous, structural triage.