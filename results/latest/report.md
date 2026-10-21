# Offline Benchmark Report

## Dataset & Split

- Dataset: `ml-100k`
- Protocol: `temporal`
- K: `10`
- Train: 70980 ratings / 849 users / 1199 items
- Validation: 9200 ratings
- Test: 9085 ratings / 848 users
- Cold-start test: 927 ratings / 94 users

## Ranking on Test

| model | ndcg_at_k | precision_at_k | recall_at_k | hit_rate_at_k | map_at_k | mrr_at_k | coverage | diversity | novelty | users |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.0919 | 0.0512 | 0.0974 | 0.3407 | 0.0508 | 0.1627 | 0.4020 | 0.5344 | 2.7753 | 766 |
| user_knn | 0.0878 | 0.0581 | 0.1001 | 0.3433 | 0.0492 | 0.1519 | 0.2010 | 0.7243 | 1.7516 | 766 |
| item_knn | 0.0816 | 0.0492 | 0.0980 | 0.3433 | 0.0451 | 0.1447 | 0.5029 | 0.7234 | 2.9892 | 766 |
| most_popular | 0.0671 | 0.0467 | 0.0843 | 0.3211 | 0.0342 | 0.1241 | 0.0601 | 0.7498 | 1.3087 | 766 |
| popularity | 0.0451 | 0.0336 | 0.0506 | 0.2363 | 0.0205 | 0.0868 | 0.0517 | 0.7429 | 2.1271 | 766 |
| funk_svd | 0.0366 | 0.0251 | 0.0350 | 0.1880 | 0.0156 | 0.0779 | 0.1485 | 0.7417 | 3.0303 | 766 |
| content_based | 0.0171 | 0.0123 | 0.0205 | 0.1084 | 0.0079 | 0.0413 | 0.3078 | 0.1108 | 3.9382 | 766 |
| global_mean | 0.0141 | 0.0134 | 0.0207 | 0.1084 | 0.0048 | 0.0203 | 0.0292 | 0.6947 | 4.1484 | 766 |
| random | 0.0073 | 0.0051 | 0.0108 | 0.0496 | 0.0029 | 0.0135 | 0.9958 | 0.7811 | 4.8201 | 766 |

## Rating Prediction on Test

| model | rmse_pooled | mae_pooled | rmse_user_mean | mae_user_mean | users | ratings |
| --- | --- | --- | --- | --- | --- | --- |
| funk_svd | 0.9775 | 0.7748 | 0.9459 | 0.8099 | 848 | 9085 |
| hybrid | 0.9792 | 0.7725 | 0.9482 | 0.8067 | 848 | 9085 |
| item_knn | 1.0423 | 0.8137 | 1.0148 | 0.8523 | 848 | 9085 |
| global_mean | 1.0614 | 0.8524 | 1.0108 | 0.8723 | 848 | 9085 |
| user_knn | 1.0783 | 0.8442 | 1.0456 | 0.8803 | 848 | 9085 |

## So sánh với baseline

| model | baseline | metric | delta_ci | p_adj | family_size | relative | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_based | popularity | ndcg_at_k | -0.0280 [-0.0364, -0.0188] | 0.0000 | 48 | -0.6207 | kém hơn |
| funk_svd | popularity | ndcg_at_k | -0.0085 [-0.0147, -0.0021] | 0.0070 | 48 | -0.1879 | kém hơn |
| global_mean | popularity | ndcg_at_k | -0.0310 [-0.0367, -0.0253] | 0.0000 | 48 | -0.6875 | kém hơn |
| hybrid | popularity | ndcg_at_k | 0.0468 [0.0340, 0.0605] | 0.0000 | 48 | 1.0371 | tốt hơn |
| item_knn | popularity | ndcg_at_k | 0.0365 [0.0234, 0.0506] | 0.0000 | 48 | 0.8104 | tốt hơn |
| most_popular | popularity | ndcg_at_k | 0.0220 [0.0116, 0.0326] | 0.0000 | 48 | 0.4871 | tốt hơn |
| random | popularity | ndcg_at_k | -0.0378 [-0.0455, -0.0300] | 0.0000 | 48 | -0.8380 | kém hơn |
| user_knn | popularity | ndcg_at_k | 0.0427 [0.0312, 0.0554] | 0.0000 | 48 | 0.9459 | tốt hơn |
| funk_svd | global_mean | rmse | -0.0648 [-0.0837, -0.0468] | 0.0000 | 8 | 0.0642 | tốt hơn |
| hybrid | global_mean | rmse | -0.0626 [-0.0808, -0.0428] | 0.0000 | 8 | 0.0620 | tốt hơn |
| item_knn | global_mean | rmse | 0.0040 [-0.0171, 0.0261] | 0.6926 | 8 | -0.0039 | không rõ |
| user_knn | global_mean | rmse | 0.0348 [0.0108, 0.0605] | 0.0002 | 8 | -0.0345 | kém hơn |

## Cold-start

| model | NDCG@K | CI low | CI high | users |
| --- | --- | --- | --- | --- |
| most_popular | 0.0267 | 0.0114 | 0.0462 | 86 |
| popularity | 0.0150 | 0.0059 | 0.0256 | 86 |
| content_based | 0.0150 | 0.0059 | 0.0256 | 86 |
| user_knn | 0.0150 | 0.0059 | 0.0256 | 86 |
| item_knn | 0.0150 | 0.0059 | 0.0256 | 86 |
| hybrid | 0.0150 | 0.0059 | 0.0256 | 86 |
| random | 0.0050 | 0.0006 | 0.0111 | 86 |
| funk_svd | 0.0043 | 0.0008 | 0.0084 | 86 |
| global_mean | 0.0012 | 0.0000 | 0.0032 | 86 |

## Efficiency

| model | fit_time_s | recommend_latency_mean_ms | recommend_latency_p95_ms |
| --- | --- | --- | --- |
| random | 0.0148 | 1.3943 | 2.4559 |
| global_mean | 0.0457 | 1.2648 | 1.4919 |
| most_popular | 0.0174 | 1.3792 | 1.6236 |
| popularity | 0.0164 | 1.4970 | 2.1656 |
| content_based | 0.0324 | 2.3291 | 3.1142 |
| user_knn | 0.0782 | 1.5677 | 1.7262 |
| item_knn | 0.1409 | 1.5946 | 2.2579 |
| funk_svd | 17.5336 | 1.3785 | 1.9179 |
| hybrid | 18.1369 | 2.9871 | 3.3727 |

## Kết luận

- `content_based`: không tốt hơn baseline; kém hơn có ý nghĩa ở ndcg_at_k.
- `funk_svd`: tốt hơn baseline có ý nghĩa ở rmse.
- `global_mean`: không tốt hơn baseline; kém hơn có ý nghĩa ở ndcg_at_k.
- `hybrid`: tốt hơn baseline có ý nghĩa ở ndcg_at_k, rmse.
- `item_knn`: tốt hơn baseline có ý nghĩa ở ndcg_at_k.
- `most_popular`: tốt hơn baseline có ý nghĩa ở ndcg_at_k.
- `random`: không tốt hơn baseline; kém hơn có ý nghĩa ở ndcg_at_k.
- `user_knn`: tốt hơn baseline có ý nghĩa ở ndcg_at_k.

## Ghi chú

- `rmse_pooled`/`mae_pooled` tính trên toàn bộ rating của tập test; `rmse_user_mean`/`mae_user_mean` là trung bình theo từng user (macro-average).
- Metric ranking là trung bình theo từng user; cột `users` là số user thực sự được đánh giá.
- `significance` dùng paired bootstrap 95% CI và Wilcoxon, hiệu chỉnh Holm–Bonferroni trong từng family (split + task); cột `family_size` là số phép so sánh trong family đó.
- Primary metric: `ndcg_at_k` cho ranking, `rmse` cho rating.
- Kết quả chỉ có ý nghĩa trong phạm vi protocol và dữ liệu đã dùng.
