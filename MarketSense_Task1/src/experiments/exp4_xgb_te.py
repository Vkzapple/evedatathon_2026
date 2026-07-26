"""Experiment 4: XGBoost + CV-safe target encoding for strong categoricals."""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from MarketSense_Task1.src.features import build_features, add_group_features

SEED = 42
N_FOLDS = 5
TE_COLS = ['latlon_key', 'neighbourhood', 'property_type']
SMOOTH = 20.0

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
print(f'base features: {X.shape[1]}, prep {time.time()-t0:.0f}s', flush=True)


def target_encode(X, X_test, y, col, skf, smooth=SMOOTH):
    """Per-class OOF target encoding: for each of 5 classes add P(class|cat)."""
    n_class = 5
    prior = np.bincount(y, minlength=n_class) / len(y)
    enc_tr = np.zeros((len(X), n_class))
    for itr, iva in skf.split(X, y):
        sub = pd.DataFrame({'c': X[col].iloc[itr].astype(str), 'y': y[itr]})
        stats = sub.groupby('c')['y'].agg(['count'])
        onehot = pd.get_dummies(sub['y']).groupby(sub['c']).sum()
        for k in range(n_class):
            if k not in onehot:
                onehot[k] = 0
        post = (onehot[range(n_class)].values + smooth * prior) / (
            stats['count'].values[:, None] + smooth)
        mapping = pd.DataFrame(post, index=stats.index, columns=range(n_class))
        va_keys = X[col].iloc[iva].astype(str)
        enc = mapping.reindex(va_keys).values
        enc = np.where(np.isnan(enc), prior, enc)
        enc_tr[iva] = enc
    # full-train encoding for test
    sub = pd.DataFrame({'c': X[col].astype(str), 'y': y})
    stats = sub.groupby('c')['y'].agg(['count'])
    onehot = pd.get_dummies(sub['y']).groupby(sub['c']).sum()
    post = (onehot[range(n_class)].values + smooth * prior) / (
        stats['count'].values[:, None] + smooth)
    mapping = pd.DataFrame(post, index=stats.index, columns=range(n_class))
    enc_te = mapping.reindex(X_test[col].astype(str)).values
    enc_te = np.where(np.isnan(enc_te), prior, enc_te)
    return enc_tr, enc_te


skf_te = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=7)
te_tr_parts, te_te_parts = [], []
for col in TE_COLS:
    a, b = target_encode(X, X_test, y, col, skf_te)
    te_tr_parts.append(pd.DataFrame(a, columns=[f'te_{col}_{k}' for k in range(5)]))
    te_te_parts.append(pd.DataFrame(b, columns=[f'te_{col}_{k}' for k in range(5)]))
X = pd.concat([X] + te_tr_parts, axis=1)
X_test = pd.concat([X_test.reset_index(drop=True)] + te_te_parts, axis=1)
print(f'with TE: {X.shape[1]} features ({time.time()-t0:.0f}s)', flush=True)

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
    model = xgb.train(params, dtr, num_boost_round=3000,
                      evals=[(dva, 'va')], early_stopping_rounds=100,
                      verbose_eval=False)
    oof[iva] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
    pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS
    print(f'fold {fold}: acc={accuracy_score(y[iva], oof[iva].argmax(1)):.4f} '
          f'best_iter={model.best_iteration} ({time.time()-t0:.0f}s)', flush=True)

oof_lbl = oof.argmax(1)
print(f'\nOOF accuracy : {accuracy_score(y, oof_lbl):.4f}')
print(f'OOF macro-F1 : {f1_score(y, oof_lbl, average="macro"):.4f}')

np.save('oof_xgb_te.npy', oof)
np.save('pred_xgb_te.npy', pred)
print(f'total {time.time()-t0:.1f}s')
