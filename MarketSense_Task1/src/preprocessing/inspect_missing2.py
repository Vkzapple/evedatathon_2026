"""Inspect the remaining ~8% rows without extracted price."""
import re
import pandas as pd

tr = pd.read_csv('train.csv')
leak = pd.read_csv('leak_train.csv')
tr = tr.merge(leak, on='id')
miss = tr[tr['leak_price'].isna()]
print('missing:', len(miss))

s = (miss['description'].fillna('') + ' ||| ' + miss['lattest comment'].fillna('')
     + ' ||| ' + miss['neighborhood_overview'].fillna(''))
pat = re.compile(r'(?i)(price|rate|cost|/night|per night|usd|\$)')
print('still has price-word:', s.str.contains(pat).mean().round(3))

cnt = 0
for txt in s:
    for m in re.finditer(r'(?i).{0,70}(?:price|rate|cost|per night|/night|\$).{0,70}',
                         str(txt)):
        frag = m.group(0).replace('\n', ' ')
        if re.search(r'(?i)(\d|one |two |three |four |five |six |seven |eight |nine |hundred|thousand)', frag):
            print('>>', frag[:150])
            cnt += 1
            break
    if cnt > 30:
        break
