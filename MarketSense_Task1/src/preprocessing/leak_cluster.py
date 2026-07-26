"""Is leaked price cleanly separable per latlon cluster / neighbourhood?"""
import re
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

tr = pd.read_csv('train.csv')
leak = pd.read_csv('leak_train.csv')
tr = tr.merge(leak, on='id')
tr['latlon_key'] = tr['latitude'].round(3).astype(str) + '_' + tr['longitude'].round(3).astype(str)

sub = tr.dropna(subset=['leak_price']).copy()
sub['logp'] = np.log1p(sub['leak_price'])

# per latlon cluster separability
tot_c, tot_n = 0, 0
accs = []
for key, s in sub.groupby('latlon_key'):
    if len(s) < 30:
        continue
    dt = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
    dt.fit(s[['logp']], s['target'])
    acc = dt.score(s[['logp']], s['target'])
    accs.append((key, len(s), acc))
    tot_c += acc * len(s); tot_n += len(s)
accs.sort(key=lambda t: t[2])
print(f'clusters>=30: {len(accs)}, weighted acc: {tot_c/tot_n:.4f}')
print('worst 5:', [(k, n, round(a, 3)) for k, n, a in accs[:5]])
print('best 5 :', [(k, n, round(a, 3)) for k, n, a in accs[-5:]])

# Maybe boundaries are GLOBAL after converting to a common currency.
# Try: price percentile within city -> tree
def normalize_city(s):
    s = str(s).lower().replace('3', 'e')
    m = re.search(r'(?:city|cty)[^a-z]*([abcd])(?![a-z])', s)
    return m.group(1) if m else 'unk'
sub['city_n'] = sub['city'].map(normalize_city)

sub['pct_city'] = sub.groupby('city_n')['logp'].rank(pct=True)
dt = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
dt.fit(sub[['pct_city']], sub['target'])
print(f'\npercentile-within-city tree acc: {dt.score(sub[["pct_city"]], sub["target"]):.4f}')
thr = sorted(t for t in dt.tree_.threshold if t > 0)
print('percentile thresholds:', [round(t, 3) for t in thr])
print('class props overall:', sub['target'].value_counts(normalize=True).sort_index().round(3).to_dict())

# check monotonic: mean rank per class
print('\nmean pct per class:', sub.groupby('target')['pct_city'].mean().round(3).to_dict())

# maybe city b contains 2 currencies: check bimodality of class-2 prices in city b
b2 = sub[(sub['city_n'] == 'b') & (sub['target'] == 2)]['leak_price']
print('\ncity b class2 price deciles:', np.percentile(b2, [10, 25, 50, 75, 90]).round(0))
b0 = sub[(sub['city_n'] == 'b') & (sub['target'] == 0)]['leak_price']
b4 = sub[(sub['city_n'] == 'b') & (sub['target'] == 4)]['leak_price']
print('city b class0 deciles:', np.percentile(b0, [10, 25, 50, 75, 90]).round(0))
print('city b class4 deciles:', np.percentile(b4, [10, 25, 50, 75, 90]).round(0))
