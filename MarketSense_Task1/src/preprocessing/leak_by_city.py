"""Check leaked price separability per city (currency differs by city)."""
import re
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

tr = pd.read_csv('train.csv')

pat = re.compile(r'price\s*info\s*:?\s*([\d,\.]+)', re.I)
def extract_price(s):
    if pd.isna(s):
        return np.nan
    m = pat.search(str(s))
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(',', '').rstrip('.'))
    except ValueError:
        return np.nan

def normalize_city(s):
    s = str(s).lower().replace('3', 'e')
    m = re.search(r'(?:city|cty)[^a-z]*([abcd])(?![a-z])', s)
    return m.group(1) if m else 'unk'

tr['price'] = tr['lattest comment'].map(extract_price)
tr['city_n'] = tr['city'].map(normalize_city)
sub = tr.dropna(subset=['price'])

for c in ['a', 'b', 'c', 'd']:
    s = sub[sub['city_n'] == c]
    print(f'--- city {c} (n={len(s)}) ---')
    print(s.groupby('target')['price'].agg(['count', 'min', 'median', 'max']).round(0))
    dt = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
    dt.fit(np.log1p(s[['price']]), s['target'])
    acc = dt.score(np.log1p(s[['price']]), s['target'])
    thr = sorted(np.expm1(t) for t in dt.tree_.threshold if t > 0)
    print(f'1-feat tree acc: {acc:.4f} | thresholds: {[round(t) for t in thr]}')
    print()
