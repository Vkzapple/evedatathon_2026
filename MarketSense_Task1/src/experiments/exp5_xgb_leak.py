"""Experiment 5: baseline features + leaked price features. 5-fold CV."""
import re
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from MarketSense_Task1.src.features import build_features, add_group_features

SEED = 42
N_FOLDS = 5

t0 = time.time()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train['target'].values

leak_tr = pd.read_csv('leak_train.csv')
leak_te = pd.read_csv('leak_test.csv')
train = train.merge(leak_tr, on='id')
test = test.merge(leak_te, on='id')

raw_all = pd.concat([train.drop(columns=['target']), test], ignore_index=True)
feats = build_features(raw_all)
feats = add_group_features(feats, raw_all)

# ---- leaked price features ----
lp = raw_all['leak_price']
feats['leak_price'] = lp
feats['leak_log'] = np.log1p(lp)
feats['leak_per_person'] = lp / feats['accommodates'].clip(lower=1)
feats['leak_per_bed'] = lp / feats['beds'].clip(lower=1)
feats['leak_per_bedroom'] = lp / feats['bedrooms'].clip(lower=1)
# percentile within city and within location cluster (train+test combined, no target)
feats['leak_pct_city'] = feats.groupby('city')['leak_log'].rank(pct=True)
feats['leak_pct_loc'] = feats.groupby('latlon_key')['leak_log'].rank(pct=True)
feats['leak_pct_neigh'] = feats.groupby('neighbourhood')['leak_log'].rank(pct=True)
# relative to cluster median
feats['leak_rel_loc'] = feats['leak_log'] - feats.groupby('latlon_key')['leak_log'].transform('median')
feats['leak_rel_city'] = feats['leak_log'] - feats.groupby('city')['leak_log'].transform('median')
feats['has_leak'] = lp.notna().astype(int)

cat_cols = ['city', 'room_type', 'property_type', 'neighbourhood',
            'host_response_time', 'latlon_key']
for c in cat_cols:
    feats[c] = feats[c].astype('category')

X = feats.iloc[:len(train)].reset_index(drop=True)
X_test = feats.iloc[len(train):].reset_index(drop=True)
print(f'features: {X.shape[1]}, prep {time.time()-t0:.0f}s', flush=True)

params = dict(
    objective='multi:softprob', num_class=5, eval_metric='mlogloss',
    learning_rate=0.05, max_depth=8, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method='hist', seed=SEED, n_jobs=-1,
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(X), 5))
pred = np.zeros((len(X_test), 5))
dtest = xgb.DMatrix(X_test, enable_categorical=True)
for fold, (itr, iva) in enumerate(skf.split(X, y)):
    dtr = xgb.DMatrix(X.iloc[itr], y[itr], enable_categorical=True)
    dva = xgb.DMatrix(X.iloc[iva], y[iva], enable_categorical=True)
    model = xgb.train(params, dtr, num_boost_round=4000,
                      evals=[(dva, 'va')], early_stopping_rounds=100,
                      verbose_eval=False)
    best = model.best_iteration + 1
    oof[iva] = model.predict(dva, iteration_range=(0, best))
    pred += model.predict(dtest, iteration_range=(0, best)) / N_FOLDS
    print(f'fold {fold}: acc={accuracy_score(y[iva], oof[iva].argmax(1)):.4f} '
          f'best_iter={model.best_iteration} ({time.time()-t0:.0f}s)', flush=True)

oof_lbl = oof.argmax(1)
print(f'\nOOF accuracy : {accuracy_score(y, oof_lbl):.4f}')
print(f'OOF macro-F1 : {f1_score(y, oof_lbl, average="macro"):.4f}')

# accuracy split by leak availability
has = X['has_leak'] == 1
print(f'acc rows WITH leak ({has.mean():.2%}): {accuracy_score(y[has], oof_lbl[has]):.4f}')
print(f'acc rows WITHOUT leak: {accuracy_score(y[~has], oof_lbl[~has]):.4f}')

imp = pd.Series(model.get_score(importance_type='gain')).sort_values(ascending=False)
print('\ntop 20 features:')
print(imp.head(20).round(1))

np.save('oof_xgb_leak.npy', oof)
np.save('pred_xgb_leak.npy', pred)
print(f'total {time.time()-t0:.0f}s')
