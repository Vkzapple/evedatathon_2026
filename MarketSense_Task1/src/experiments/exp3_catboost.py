"""Experiment 3: CatBoost with native categorical + text features, 5-fold CV."""
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from MarketSense_Task1.src.features import build_features, add_group_features

SEED = 42
N_FOLDS = 5

t0 = time.time()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train['target'].values

raw_all = pd.concat([train.drop(columns=['target']), test], ignore_index=True)
feats = build_features(raw_all)
feats = add_group_features(feats, raw_all)

CAT_COLS = ['city', 'room_type', 'property_type', 'neighbourhood',
            'host_response_time', 'latlon_key']
TEXT_COLS = ['txt_name', 'txt_desc', 'txt_comment']
feats['txt_name'] = raw_all['name'].fillna('').astype(str)
feats['txt_desc'] = raw_all['description'].fillna('').astype(str)
feats['txt_comment'] = raw_all['lattest comment'].fillna('').astype(str)
for c in CAT_COLS:
    feats[c] = feats[c].astype(str)

num_cols = [c for c in feats.columns if c not in CAT_COLS + TEXT_COLS]
feats[num_cols] = feats[num_cols].astype(np.float32)

X = feats.iloc[:len(train)].reset_index(drop=True)
X_test = feats.iloc[len(train):].reset_index(drop=True)
print(f'features: {X.shape[1]}, prep took {time.time()-t0:.1f}s', flush=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(X), 5))
pred = np.zeros((len(X_test), 5))
test_pool = Pool(X_test, cat_features=CAT_COLS, text_features=TEXT_COLS)
for fold, (itr, iva) in enumerate(skf.split(X, y)):
    tr_pool = Pool(X.iloc[itr], y[itr], cat_features=CAT_COLS, text_features=TEXT_COLS)
    va_pool = Pool(X.iloc[iva], y[iva], cat_features=CAT_COLS, text_features=TEXT_COLS)
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.08, depth=8, l2_leaf_reg=3.0,
        loss_function='MultiClass', eval_metric='Accuracy',
        random_seed=SEED, early_stopping_rounds=150, verbose=0,
    )
    model.fit(tr_pool, eval_set=va_pool)
    oof[iva] = model.predict_proba(va_pool)
    pred += model.predict_proba(test_pool) / N_FOLDS
    print(f'fold {fold}: acc={accuracy_score(y[iva], oof[iva].argmax(1)):.4f} '
          f'best_iter={model.get_best_iteration()} ({time.time()-t0:.0f}s)', flush=True)

oof_lbl = oof.argmax(1)
print(f'\nOOF accuracy : {accuracy_score(y, oof_lbl):.4f}')
print(f'OOF macro-F1 : {f1_score(y, oof_lbl, average="macro"):.4f}')

np.save('oof_cat.npy', oof)
np.save('pred_cat.npy', pred)
print(f'total {time.time()-t0:.1f}s')
