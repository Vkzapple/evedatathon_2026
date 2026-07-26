"""Comprehensive price extraction from description + lattest comment.

Findings so far:
- 'Price info: X' in lattest comment (33% rows), X == description price (corr 1.0)
- description has price mentions in ~55% rows -> broaden patterns
- price scale differs per city (currency) -> per-city thresholds
"""
import re
import numpy as np
import pandas as pd

tr = pd.read_csv('train.csv')

# Look at how price appears in description
s = tr['description'].fillna('')
price_rows = s[s.str.contains(r'(?i)price|/night|per night|a night')].head(40)
for txt in price_rows:
    for m in re.finditer(r'(?i).{0,50}(?:price|night).{0,50}', txt):
        frag = m.group(0).replace('\n', ' ')
        if re.search(r'\d', frag):
            print('>>', frag)
    print('---')
