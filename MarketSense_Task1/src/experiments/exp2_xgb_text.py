"""Experiment 2: XGBoost + TF-IDF/SVD text features, 5-fold CV."""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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

# ---- text: combine listing-authored fields, TF-IDF -> SVD ----
text = (raw_all['name'].fillna('') + ' ' + raw_all['description'].fillna('')
        + ' ' + raw_all['neighborhood_overview'].fillna(''))
tfv = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=3,
                      sublinear_tf=True, strip_accents='unicode')
Xt = tfv.fit_transform(text)
svd = TruncatedSVD(n_components=128, random_state=SEED)
Xsvd = svd.fit_transform(Xt).astype(np.float32)
print(f'tfidf {Xt.shape} -> svd 128, explained {svd.explained_variance_ratio_.sum():.3f}, {time.time()-t0:.0f}s')

for i in range(Xsvd.shape[1]):
    feats[f'svd_{i}'] = Xsvd[:, i]

cat_cols = ['city', 'room_type', 'property_type', 'neighbourhood',
            'host_response_time', 'latlon_key']
for c in cat_cols:
    feats[c] = feats[c].astype('category')

X = feats.iloc[:len(train)].reset_index(drop=True)
X_test = feats.iloc[len(train):].reset_index(drop=True)
print(f'features: {X.shape[1]}, prep took {time.time()-t0:.1f}s')

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

np.save('oof_xgb_text.npy', oof)
np.save('pred_xgb_text.npy', pred)
print(f'total {time.time()-t0:.1f}s')
