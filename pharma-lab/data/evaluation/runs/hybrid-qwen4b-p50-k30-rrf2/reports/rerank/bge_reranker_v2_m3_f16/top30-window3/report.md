# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 87.95% |
| Hit@5 | 92.05% |
| Hit@10 | 95.81% |
| Hit@30 | 99.17% |
| nDCG@10 | 0.8367 |
| MRR@10 | 0.8044 |
| MRR@30 | 0.8065 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 72.10% |
| Complete-evidence rate@3 | 48.60% |
| Recall@5 | 85.00% |
| Complete-evidence rate@5 | 71.40% |
| Recall@10 | 95.60% |
| Complete-evidence rate@10 | 91.20% |
| Recall@30 | 99.90% |
| Complete-evidence rate@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 89.30% | 92.70% | 96.30% | 99.00% | 0.8001 | 0.7456 | 0.7472 |
| formulary | 5000 | 84.84% | 90.82% | 95.82% | 99.20% | 0.8065 | 0.7572 | 0.7594 |
| brand_product_qa | 2500 | 97.60% | 98.60% | 99.24% | 99.72% | 0.9555 | 0.9431 | 0.9435 |
| multi_intent | 500 | 95.60% | 98.60% | 100.00% | 100.00% | 0.8076 | 0.8906 | 0.8906 |
| noisy_confuser | 500 | 48.40% | 55.80% | 69.20% | 94.80% | 0.4912 | 0.4291 | 0.4447 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9919 | 0.9890 | 0.9890 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 74.11% | 84.97% | 92.49% | 99.90% | 0.6991 | 0.6263 | 0.6311 |
| hard | 568 | 94.89% | 97.71% | 99.65% | 99.65% | 0.8201 | 0.8911 | 0.8911 |
| medium | 8447 | 89.10% | 92.49% | 95.94% | 99.05% | 0.8538 | 0.8193 | 0.8212 |
