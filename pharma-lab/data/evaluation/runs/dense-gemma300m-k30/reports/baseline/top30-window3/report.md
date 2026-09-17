# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 67.49% |
| Hit@5 | 75.55% |
| Hit@10 | 84.00% |
| Hit@30 | 90.98% |
| nDCG@10 | 0.6404 |
| MRR@10 | 0.5866 |
| MRR@30 | 0.5910 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 55.90% |
| Complete-evidence rate@3 | 24.80% |
| Recall@5 | 65.80% |
| Complete-evidence rate@5 | 39.40% |
| Recall@10 | 76.20% |
| Complete-evidence rate@10 | 55.40% |
| Recall@30 | 86.10% |
| Complete-evidence rate@30 | 73.20% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 62.90% | 76.30% | 89.70% | 97.80% | 0.5929 | 0.4961 | 0.5013 |
| formulary | 5000 | 61.00% | 70.72% | 81.10% | 89.90% | 0.5884 | 0.5178 | 0.5234 |
| brand_product_qa | 2500 | 82.88% | 86.36% | 90.44% | 93.44% | 0.7998 | 0.7660 | 0.7679 |
| multi_intent | 500 | 87.00% | 92.20% | 97.00% | 99.00% | 0.6363 | 0.7261 | 0.7275 |
| noisy_confuser | 500 | 40.40% | 51.00% | 61.80% | 75.80% | 0.4050 | 0.3384 | 0.3466 |
| patient_natural | 500 | 72.20% | 76.20% | 78.60% | 83.00% | 0.6970 | 0.6680 | 0.6708 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 78.48% | 83.55% | 88.53% | 94.01% | 0.7396 | 0.6923 | 0.6957 |
| hard | 568 | 85.74% | 90.85% | 95.25% | 97.36% | 0.6442 | 0.7183 | 0.7198 |
| medium | 8447 | 64.98% | 73.59% | 82.72% | 90.20% | 0.6285 | 0.5655 | 0.5702 |
