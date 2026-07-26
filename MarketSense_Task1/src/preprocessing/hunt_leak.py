"""Hunt for leaked price signal in text/other columns."""
import re
import numpy as np
import pandas as pd

tr = pd.read_csv('train.csv')

# 1) Search all text columns for price-like patterns
pat = re.compile(r'(?:\$|usd|rp|idr|price|/night|per night|a night)', re.I)
for c in ['name', 'description', 'neighborhood_overview', 'host_about', 'lattest comment']:
    s = tr[c].fillna('')
    hit = s.str.contains(pat)
    print(f'{c:25s} price-pattern hit rate: {hit.mean():.3f}')

# 2) Show examples of hits in each column
for c in ['name', 'description', 'lattest comment']:
    s = tr[c].fillna('')
    hits = tr.loc[s.str.contains(pat), [c, 'target']].head(3)
    for _, row in hits.iterrows():
        txt = str(row[c])
        m = pat.search(txt)
        lo = max(0, m.start() - 60)
        print(f'[{c}] target={row["target"]}: ...{txt[lo:m.start()+80]}...')
    print()

# 3) dollar amounts specifically
dollar = re.compile(r'\$\s?(\d{2,5})')
for c in ['name', 'description', 'lattest comment', 'neighborhood_overview', 'host_about']:
    s = tr[c].fillna('')
    ex = s.str.extract(dollar, expand=False)
    n = ex.notna().sum()
    if n:
        sub = pd.DataFrame({'amt': pd.to_numeric(ex), 'target': tr['target']}).dropna()
        print(f'{c}: {n} rows with $amount; mean amount per target:')
        print(sub.groupby('target')['amt'].agg(['count', 'mean', 'median']).round(0))
        print()
