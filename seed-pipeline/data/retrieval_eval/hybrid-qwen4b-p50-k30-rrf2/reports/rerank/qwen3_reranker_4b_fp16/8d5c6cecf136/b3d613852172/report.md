# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 88.51% |
| Hit@5 | 92.97% |
| Hit@10 | 96.98% |
| Hit@30 | 99.09% |
| MRR | 0.8060 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 53.60% |
| Multi-all-hit@3 | 28.40% |
| Multi-section Recall@5 | 69.20% |
| Multi-all-hit@5 | 50.00% |
| Multi-section Recall@10 | 87.00% |
| Multi-all-hit@10 | 77.20% |
| Multi-section Recall@30 | 99.80% |
| Multi-all-hit@30 | 99.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 97.44% | 98.28% | 99.04% | 99.72% | 0.9484 |
| chunk_level_retrieval | 1000 | 87.40% | 92.30% | 96.20% | 99.10% | 0.7930 |
| formulary | 5000 | 87.44% | 92.78% | 97.38% | 99.02% | 0.7686 |
| multi_intent | 500 | 78.80% | 88.40% | 96.80% | 100.00% | 0.6490 |
| noisy_confuser | 500 | 55.00% | 67.20% | 81.40% | 94.80% | 0.4724 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9850 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 85.69% | 91.98% | 97.56% | 99.80% | 0.7611 |
| hard | 568 | 80.28% | 88.73% | 96.83% | 99.65% | 0.6804 |
| medium | 8447 | 89.39% | 93.37% | 96.92% | 98.97% | 0.8197 |
