# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 83.70% |
| Hit@5 | 90.24% |
| Hit@10 | 95.67% |
| Hit@30 | 99.09% |
| MRR | 0.7242 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 60.80% |
| Multi-all-hit@3 | 29.40% |
| Multi-section Recall@5 | 78.10% |
| Multi-all-hit@5 | 59.20% |
| Multi-section Recall@10 | 93.30% |
| Multi-all-hit@10 | 87.00% |
| Multi-section Recall@30 | 99.80% |
| Multi-all-hit@30 | 99.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 67.50% | 82.10% | 92.80% | 99.10% | 0.5330 |
| formulary | 5000 | 82.78% | 90.12% | 95.84% | 99.02% | 0.6890 |
| brand_product_qa | 2500 | 93.44% | 96.08% | 98.48% | 99.72% | 0.8877 |
| multi_intent | 500 | 92.20% | 97.00% | 99.60% | 100.00% | 0.6809 |
| noisy_confuser | 500 | 51.80% | 62.00% | 77.40% | 94.80% | 0.4396 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9683 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 89.44% | 93.91% | 96.95% | 99.80% | 0.6813 |
| hard | 568 | 91.90% | 96.13% | 98.77% | 99.65% | 0.7057 |
| medium | 8447 | 82.48% | 89.42% | 95.31% | 98.97% | 0.7304 |
