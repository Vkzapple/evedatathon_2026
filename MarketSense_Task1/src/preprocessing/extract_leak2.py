"""Comprehensive leaked-price extractor + coverage measurement."""
import re
import numpy as np
import pandas as pd

tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')

NUM = r'\$?\s*([\d.,]+)\s*[kK]?'

PATTERNS = [
    re.compile(r'price\s*info\s*:?\s*([\d.,]+)\s*([kK])?', re.I),
    re.compile(r'price\s*(?:is)?\s*(?:only)?\s*:\s*\$?\s*([\d.,]+)\s*([kK])?', re.I),
    re.compile(r'(?:the\s+)?price\s+is\s+only\s+\$?\s*([\d.,]+)\s*([kK])?', re.I),
    re.compile(r'worth\s+it,?\s+only\s+\$?\s*([\d.,]+)\s*([kK])?\s*per\s*night', re.I),
    re.compile(r'value\s+for\s+money\s+at\s+\$?\s*([\d.,]+)\s*([kK])?\s*per\s*night', re.I),
    re.compile(r'\$?\s*([\d.,]+)\s*([kK])?\s+is\s+the\s+price', re.I),
    re.compile(r'price\s*:?\s*\$?\s*([\d.,]+)\s*([kK])?\s*!', re.I),
    re.compile(r'only\s+\$\s*([\d.,]+)\s*([kK])?\s*!', re.I),
    re.compile(r'only\s+\$?\s*([\d.,]+)\s*([kK])?\s*per\s*night', re.I),
    re.compile(r'\$\s*([\d.,]+)\s*([kK])?\s*(?:per|/|a)\s*night', re.I),
]


def parse_num(numstr: str, ksuf) -> float:
    """Handle '2,057' '173.00' '13.377' '0.189'+k '16.548'+k formats."""
    s = numstr.strip().rstrip('.').rstrip(',')
    has_k = bool(ksuf)
    # remove commas used as thousands sep
    if ',' in s and '.' in s:
        # assume , thousands . decimal
        s = s.replace(',', '')
        val = float(s)
    elif ',' in s:
        parts = s.split(',')
        if all(len(p) == 3 for p in parts[1:]):
            val = float(s.replace(',', ''))
        else:
            val = float(s.replace(',', '.'))
    elif '.' in s:
        parts = s.split('.')
        if len(parts[-1]) == 2 and len(parts) == 2:      # 173.00 decimal
            val = float(s)
        elif all(len(p) == 3 for p in parts[1:]):        # 13.377 thousands
            if has_k:
                val = float(s)                            # 16.548k -> 16.548*1000
            else:
                val = float(s.replace('.', ''))
        else:
            val = float(s)
    else:
        val = float(s)
    if has_k:
        val *= 1000.0
    return val


def extract_price(row) -> float:
    for col in ('lattest comment', 'description', 'neighborhood_overview', 'name'):
        txt = row.get(col)
        if pd.isna(txt):
            continue
        txt = str(txt)
        for pat in PATTERNS:
            m = pat.search(txt)
            if m:
                try:
                    v = parse_num(m.group(1), m.group(2) if m.lastindex >= 2 else None)
                    if v > 0:
                        return v
                except (ValueError, IndexError):
                    continue
    return np.nan


for name, df in [('train', tr), ('test', te)]:
    df['leak_price'] = df.apply(extract_price, axis=1)
    print(f'{name}: coverage = {df["leak_price"].notna().mean():.4f}')

sub = tr.dropna(subset=['leak_price'])
print('\nper-target stats (all cities mixed):')
print(sub.groupby('target')['leak_price'].agg(['count', 'median']).round(0))

# per-city separability with log price
def normalize_city(s):
    s = str(s).lower().replace('3', 'e')
    m = re.search(r'(?:city|cty)[^a-z]*([abcd])(?![a-z])', s)
    return m.group(1) if m else 'unk'

sub = sub.copy()
sub['city_n'] = sub['city'].map(normalize_city)
from sklearn.tree import DecisionTreeClassifier
total_correct, total_n = 0, 0
for c in sorted(sub['city_n'].unique()):
    s = sub[sub['city_n'] == c]
    if len(s) < 50:
        continue
    dt = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
    Xp = np.log1p(s[['leak_price']])
    dt.fit(Xp, s['target'])
    acc = dt.score(Xp, s['target'])
    thr = sorted(np.expm1(t) for t in dt.tree_.threshold if t > 0)
    print(f'city {c}: n={len(s)} tree-acc={acc:.4f} thresholds={[round(t) for t in thr]}')
    total_correct += acc * len(s)
    total_n += len(s)
print(f'\nweighted per-city acc on leaked rows: {total_correct/total_n:.4f}')

tr[['id', 'leak_price']].to_csv('leak_train.csv', index=False)
te[['id', 'leak_price']].to_csv('leak_test.csv', index=False)
