# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 86.88% |
| Hit@5 | 90.91% |
| Hit@10 | 95.04% |
| Hit@30 | 99.09% |
| MRR | 0.7982 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 64.50% |
| Multi-all-hit@3 | 36.60% |
| Multi-section Recall@5 | 75.60% |
| Multi-all-hit@5 | 56.20% |
| Multi-section Recall@10 | 89.50% |
| Multi-all-hit@10 | 81.20% |
| Multi-section Recall@30 | 99.80% |
| Multi-all-hit@30 | 99.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 97.60% | 98.60% | 99.24% | 99.72% | 0.9435 |
| chunk_level_retrieval | 1000 | 89.40% | 92.80% | 96.40% | 99.10% | 0.7482 |
| formulary | 5000 | 83.00% | 88.90% | 94.46% | 99.02% | 0.7451 |
| multi_intent | 500 | 92.40% | 95.00% | 97.80% | 100.00% | 0.8656 |
| noisy_confuser | 500 | 48.40% | 55.60% | 69.40% | 94.80% | 0.4437 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9890 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 72.18% | 81.93% | 89.75% | 99.80% | 0.6178 |
| hard | 568 | 92.08% | 94.54% | 97.71% | 99.65% | 0.8691 |
| medium | 8447 | 88.24% | 91.71% | 95.48% | 98.97% | 0.8144 |
