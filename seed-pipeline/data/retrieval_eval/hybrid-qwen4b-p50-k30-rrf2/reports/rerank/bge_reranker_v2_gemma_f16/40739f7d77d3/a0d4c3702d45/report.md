# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 84.55% |
| Hit@5 | 90.19% |
| Hit@10 | 95.30% |
| Hit@30 | 99.09% |
| MRR | 0.7570 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 64.90% |
| Multi-all-hit@3 | 38.40% |
| Multi-section Recall@5 | 78.20% |
| Multi-all-hit@5 | 59.80% |
| Multi-section Recall@10 | 89.80% |
| Multi-all-hit@10 | 80.40% |
| Multi-section Recall@30 | 99.80% |
| Multi-all-hit@30 | 99.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 95.72% | 98.00% | 99.28% | 99.72% | 0.9127 |
| chunk_level_retrieval | 1000 | 83.20% | 90.20% | 95.00% | 99.10% | 0.7129 |
| formulary | 5000 | 80.24% | 87.36% | 94.36% | 99.02% | 0.6937 |
| multi_intent | 500 | 91.40% | 96.60% | 99.20% | 100.00% | 0.7846 |
| noisy_confuser | 500 | 52.20% | 63.20% | 76.80% | 94.80% | 0.4364 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9940 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 69.44% | 78.38% | 89.64% | 99.80% | 0.6029 |
| hard | 568 | 91.55% | 96.30% | 98.77% | 99.65% | 0.7997 |
| medium | 8447 | 85.84% | 91.16% | 95.73% | 98.97% | 0.7721 |
