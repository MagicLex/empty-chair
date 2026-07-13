# empty-chair bias / confound audit

Pre-publication check. No protected attribute is inferred. The question: does the score track concealment, or a legitimate corporate/property structure that correlates with both the labels and certain business communities?

## 1. SIC concentration in the top 1%

| SIC | activity | share of top 1% | lift vs population |
|---|---|--:|--:|
| 68 | real estate | 43.4% | 4.0x |
| 47 | retail | 13.8% | 1.8x |
| 62 | IT | 8.6% | 1.7x |
| 46 | other | 7.7% | 2.3x |
| 41 | construction | 3.0% | 0.7x |
| 58 | other | 2.4% | 2.9x |
| 64 | financial holding / trusts | 2.2% | 0.6x |
| 70 | head office / management consulting | 1.7% | 0.3x |
| 59 | other | 0.9% | 0.7x |
| 56 | other | 0.9% | 0.2x |

## 2. Structural composition: top 1% vs whole population

| trait | top 1% | population | positives (labels) |
|---|--:|--:|--:|
| psc_absent | 0.7% | 6.2% | 7.9% |
| psc_silence | 2.4% | 6.5% | 7.6% |
| psc_corporate_only | 0.6% | 5.9% | 17.2% |
| psc_foreign_corporate | 0.5% | 1.0% | 5.3% |
| is_mill_address | 46.1% | 28.1% | 29.0% |
| is_holding_sic | 15.1% | 12.1% | 25.1% |
| accounts_dormant | 58.6% | 37.2% | 36.7% |
| active | 99.3% | 98.0% | 97.6% |

## 3. Confound isolation (mean score)

- holding-SIC companies: 0.116  vs  non-holding: 0.118
- foreign-corporate PSC: 0.105  vs  none: 0.118

## Reading

If the top 1% is dominated by real-estate / holding SIC at high lift, and holding-SIC and foreign-corporate alone move the mean score a lot, then the model is substantially a **corporate-structure detector**: it finds SPV / holding / multinational vehicles, which include both genuine concealment and ordinary legitimate structuring (property developers, REITs, group treasuries). It cannot tell intent apart. Publishing named individuals as concealment-linked on this basis is not defensible; the honest use is anonymous, structural triage.