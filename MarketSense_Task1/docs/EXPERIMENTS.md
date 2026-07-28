# Experiment Log - Task 1 MarketSense

Target: multiclass price class 0–4 (imbalanced: class 2 = 39%, class 0 = 6%).
Metric lokal: OOF accuracy + macro-F1 (metrik LB belum dikonfirmasi; accuracy diasumsikan).
CV: StratifiedKFold 5-fold, seed 42.

| Version | Model | Features | CV Acc | CV Macro-F1 | LB | Notes |
|---------|-------|----------|--------|-------------|----|-------|
| v1 | XGBoost hist, depth 8, lr 0.05 | 94 tabular (cleaned cats, amenities, dates, host/loc aggregates) | 0.6173 | 0.5747 | | latlon_key fitur terkuat; amenities premium (dishwasher, hot tub, pool) penting |
| v2 | XGBoost + TF-IDF(1-2gram, 60k) -> SVD 128 | 94 + 128 text | ~0.607 (fold0, stopped) | | | Text SVD MENURUNKAN skor — noise > signal; dihentikan |
| v3 | CatBoost native text features | tabular + raw text cols | dihentikan | | | Dihentikan atas permintaan; fokus submit v1 dulu |
| v5 | XGBoost + leaked price (extractor v2, coverage 92.0%) | 105 (94 + 11 leak) | 0.8697 | 0.8601 | | Harga bocor di teks (description/comment/dll); leak_pct_city fitur terkuat |
| v5b | XGBoost + leaked price (extractor v3, coverage 95.3%) | 105 | 0.8843 | 0.8769 | | Tambah pola spelled-out numbers; acc rows-with-leak 0.9025, rows-without 0.518 |
| v5c | XGBoost + leaked price (extractor v3 fixed, coverage 99.0%) | 105 | 0.9016 | 0.8951 | | Fix regex NUMWORD (nineteen match sebelum nine); suspect mis-parse 276→74; coverage 95.3%→99.0%; acc rows-with-leak 0.9056 |

## Status submission
- `submission.csv` digenerate dari `pred_xgb_leak.npy` (v5c: leak extractor v3 fixed, coverage 99.0%, OOF acc 0.9016).
- `task1.ipynb` = notebook berisi pipeline v1 lengkap (auto-detect `/kaggle/input`) — **BELUM diupdate ke pipeline leak v5c**.

## Insight EDA
- Kolom `city` dan `room_type` sengaja dibuat noisy (typo, case, `3`->`e`, spasi). Normalisasi regex → city {a,b,c,d}, room {entire, private, shared, hotel}.
- `calendar_updated` 100% kosong → drop.
- Hanya ~845 pasangan lat/lon unik → lokasi adalah cluster; `latlon_key` = kategorikal kuat.
- 70% host_id test ada di train → host aggregate features aman & berguna.
- `estimated_occupancy_l365d` & `number_of_reviews` turun monotonik untuk kelas 3-4 (listing mahal lebih jarang dibooking).
- Class distribution seragam antar city → city bukan pembeda target secara langsung.
