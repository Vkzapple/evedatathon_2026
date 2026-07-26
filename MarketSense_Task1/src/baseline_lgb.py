"""Local baseline: LightGBM with 5-fold Stratified CV on tabular features."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
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

cat_cols = ['city', 'room_type', 'property_type', 'neighbourhood',
            'host_response_time', 'latlon_key']
for c in cat_cols:
    feats[c] = feats[c].astype('category')

X = feats.iloc[:len(train)].reset_index(drop=True)
X_test = feats.iloc[len(train):].reset_index(drop=True)
print(f'features: {X.shape[1]}, prep took {time.time()-t0:.1f}s')

params = dict(
    objective='multiclass', num_class=5, metric='multi_logloss',
    learning_rate=0.05, num_leaves=127, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, seed=SEED, verbosity=-1, n_jobs=-1,
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(X), 5))
pred = np.zeros((len(X_test), 5))
for fold, (itr, iva) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[itr], y[itr])
    dva = lgb.Dataset(X.iloc[iva], y[iva])
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[iva] = model.predict(X.iloc[iva], num_iteration=model.best_iteration)
    pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    acc = accuracy_score(y[iva], oof[iva].argmax(1))
    print(f'fold {fold}: acc={acc:.4f} best_iter={model.best_iteration}')

oof_lbl = oof.argmax(1)
print(f'\nOOF accuracy : {accuracy_score(y, oof_lbl):.4f}')
print(f'OOF macro-F1 : {f1_score(y, oof_lbl, average="macro"):.4f}')
print(f'OOF weighted-F1: {f1_score(y, oof_lbl, average="weighted"):.4f}')

imp = pd.Series(model.feature_importance('gain'), index=X.columns)
print('\nTop 25 features:')
print(imp.sort_values(ascending=False).head(25).round(0))

np.save('oof_lgb.npy', oof)
np.save('pred_lgb.npy', pred)
print(f'\ntotal {time.time()-t0:.1f}s')
