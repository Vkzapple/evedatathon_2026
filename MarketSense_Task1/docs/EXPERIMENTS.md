# Experiment Log - Task 1 MarketSense

Target: multiclass price class 0–4 (imbalanced: class 2 = 39%, class 0 = 6%).
Metric lokal: OOF accuracy + macro-F1 (metrik LB belum dikonfirmasi; accuracy diasumsikan).
CV: StratifiedKFold 5-fold, seed 42.

| Version | Model | Features | CV Acc | CV Macro-F1 | LB | Notes |
|---------|-------|----------|--------|-------------|----|-------|
| v1 | XGBoost hist, depth 8, lr 0.05 | 94 tabular (cleaned cats, amenities, dates, host/loc aggregates) | 0.6173 | 0.5747 | | latlon_key fitur terkuat; amenities premium (dishwasher, hot tub, pool) penting |
| v2 | XGBoost + TF-IDF(1-2gram, 60k) -> SVD 128 | 94 + 128 text | ~0.607 (fold0, stopped) | | | Text SVD MENURUNKAN skor — noise > signal; dihentikan |
| v3 | CatBoost native text features | tabular + raw text cols | dihentikan | | | Dihentikan atas permintaan; fokus submit v1 dulu |

## Status submission
- `submission.csv` digenerate dari `pred_xgb.npy` (v1, rata-rata probabilitas 5 fold, argmax).
- `task1.ipynb` = notebook berisi pipeline v1 lengkap (auto-detect `/kaggle/input`).

## Insight EDA
- Kolom `city` dan `room_type` sengaja dibuat noisy (typo, case, `3`->`e`, spasi). Normalisasi regex → city {a,b,c,d}, room {entire, private, shared, hotel}.
- `calendar_updated` 100% kosong → drop.
- Hanya ~845 pasangan lat/lon unik → lokasi adalah cluster; `latlon_key` = kategorikal kuat.
- 70% host_id test ada di train → host aggregate features aman & berguna.
- `estimated_occupancy_l365d` & `number_of_reviews` turun monotonik untuk kelas 3-4 (listing mahal lebih jarang dibooking).
- Class distribution seragam antar city → city bukan pembeda target secara langsung.
