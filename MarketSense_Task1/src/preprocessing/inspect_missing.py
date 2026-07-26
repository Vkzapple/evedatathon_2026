"""Inspect rows WITHOUT extracted price — is there a hidden price form?"""
import re
import numpy as np
import pandas as pd

tr = pd.read_csv('train.csv')
leak = pd.read_csv('leak_train.csv')
tr = tr.merge(leak, on='id')
miss = tr[tr['leak_price'].isna()]
print('rows without price:', len(miss), f'({len(miss)/len(tr):.1%})')

# any digits near key words in their description?
s = miss['description'].fillna('')
has_num = s.str.contains(r'\d')
print('desc has any digit:', has_num.mean().round(3))

# sample descriptions ends (price often appended at end)
for txt in s[s.str.len() > 50].sample(15, random_state=1):
    tail = str(txt).replace('\n', ' ')[-120:]
    print('END>>', tail)
print()
# sample comment ends
c = miss['lattest comment'].fillna('')
for txt in c[c.str.len() > 50].sample(15, random_state=2):
    tail = str(txt).replace('\n', ' ')[-120:]
    print('CMT>>', tail)
print()
# name endings
n = miss['name'].fillna('')
for txt in n.sample(15, random_state=3):
    print('NAME>>', str(txt)[-80:])
