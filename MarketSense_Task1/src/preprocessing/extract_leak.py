"""Extract leaked 'Price info: X' and find the class boundaries."""
import re
import numpy as np
import pandas as pd

tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')

pat = re.compile(r'price\s*info\s*:?\s*([\d,\.]+)', re.I)

def extract_price(s):
    if pd.isna(s):
        return np.nan
    m = pat.search(str(s))
    if not m:
        return np.nan
    num = m.group(1).replace(',', '').rstrip('.')
    try:
        return float(num)
    except ValueError:
        return np.nan

tr['leak_price'] = tr['lattest comment'].map(extract_price)
te['leak_price'] = te['lattest comment'].map(extract_price)
print('train leak coverage:', tr['leak_price'].notna().mean().round(4))
print('test  leak coverage:', te['leak_price'].notna().mean().round(4))

sub = tr.dropna(subset=['leak_price'])
print('\nprice stats per target:')
print(sub.groupby('target')['leak_price'].agg(['count', 'min', 'max', 'mean', 'median']).round(0))

# Are classes separated by clean price thresholds?
print('\nquantile check — price ranges per class:')
for t in range(5):
    p = sub.loc[sub['target'] == t, 'leak_price']
    print(f'class {t}: p1={p.quantile(.01):.0f} p25={p.quantile(.25):.0f} p50={p.median():.0f} p75={p.quantile(.75):.0f} p99={p.quantile(.99):.0f}')

# check overlap: sort by price, see if target is monotonic
sub2 = sub[['leak_price', 'target']].sort_values('leak_price')
# accuracy of best simple thresholds via decision tree on 1 feature
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
dt.fit(sub2[['leak_price']], sub2['target'])
acc = dt.score(sub2[['leak_price']], sub2['target'])
print(f'\n1-feature decision tree (5 leaves) accuracy on leaked rows: {acc:.4f}')
thr = sorted(t for t in dt.tree_.threshold if t > 0)
print('thresholds:', [round(t, 1) for t in thr])

# Also check description for price mentions
pat2 = re.compile(r'(?:only|price|for)\s*\$?\s*([\d,]{3,10})\s*(?:per night|/night|a night)', re.I)
def extract_price2(s):
    if pd.isna(s):
        return np.nan
    m = pat2.search(str(s))
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(',', ''))
    except ValueError:
        return np.nan
tr['desc_price'] = tr['description'].map(extract_price2)
print('\ndesc price coverage:', tr['desc_price'].notna().mean().round(4))
both = tr.dropna(subset=['leak_price', 'desc_price'])
if len(both):
    print('corr leak vs desc price:', both['leak_price'].corr(both['desc_price']).round(3))
    print((both['leak_price'] == both['desc_price']).mean())
