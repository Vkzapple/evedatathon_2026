"""Leaked price extractor v3 — numeric + spelled-out numbers, all pattern variants."""
import re
import numpy as np
import pandas as pd

# ---------- number-word parser ----------
UNITS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
SCALES = {'hundred': 100, 'thousand': 1000, 'million': 1000000}
NUMWORD = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|and|[-\s,])+'


def words_to_num(text: str) -> float:
    tokens = re.split(r'[\s,-]+', text.lower())
    total, current = 0, 0
    seen = False
    for tok in tokens:
        if tok in UNITS:
            current += UNITS[tok]; seen = True
        elif tok in TENS:
            current += TENS[tok]; seen = True
        elif tok == 'hundred':
            current = max(current, 1) * 100; seen = True
        elif tok in ('thousand', 'million'):
            total += max(current, 1) * SCALES[tok]; current = 0; seen = True
        elif tok in ('and', ''):
            continue
        else:
            break
    return float(total + current) if seen else np.nan


NUMG = r'\$?\s*([\d.,]+)\s*([kK])?'
PATTERNS = [
    # explicit price/rate labels
    rf'price\s*info\s*:?\s*{NUMG}',
    rf'(?:price|rate)\s*(?:is)?\s*(?:only)?\s*:\s*{NUMG}',
    rf'(?:the\s+)?(?:price|rate)\s+is\s+only\s+{NUMG}',
    rf'special\s+rate\s+{NUMG}\s+for\s+this\s+week',
    rf'worth\s+it,?\s+only\s+{NUMG}\s*per\s*night',
    rf'value\s+for\s+money\s+at\s+{NUMG}\s*(?:per\s*night)?',
    rf'{NUMG}\s+is\s+the\s+price',
    rf'it\s+costs?\s+{NUMG}',
    rf'(?:price|rate)\s*:?\s*{NUMG}\s*!',
    rf'only\s+\$\s*([\d.,]+)\s*([kK])?\s*!',
    rf'worth\s+the\s+price\s+only\s+{NUMG}\s*!?',
    rf'only\s+([\d.,]+)\s*([kK])?\s*!',
    rf'only\s+{NUMG}\s*per\s*night',
    rf'\$\s*([\d.,]+)\s*([kK])?\s*(?:per|/|a)\s*night',
]
PATTERNS = [re.compile(p, re.I) for p in PATTERNS]

WORD_PATTERNS = [
    re.compile(rf'(?:price|rate)\s*:?\s*({NUMWORD})\s*!', re.I),
    re.compile(rf'(?:price|rate)\s*:\s*({NUMWORD})', re.I),
    re.compile(rf'(?:the\s+)?(?:price|rate)\s+is\s+only\s+({NUMWORD})', re.I),
    re.compile(rf'({NUMWORD})\s+is\s+the\s+(?:price|rate)', re.I),
    re.compile(rf'it\s+costs?\s+({NUMWORD})', re.I),
    re.compile(rf'special\s+rate\s+({NUMWORD})\s+for', re.I),
    re.compile(rf'worth\s+it,?\s+only\s+({NUMWORD})\s+per\s+night', re.I),
    re.compile(rf'value\s+for\s+money\s+at\s+({NUMWORD})\s+per\s+night', re.I),
]


def parse_num(numstr: str, ksuf) -> float:
    s = numstr.strip().rstrip('.').rstrip(',')
    has_k = bool(ksuf)
    try:
        if ',' in s and '.' in s:
            val = float(s.replace(',', ''))
        elif ',' in s:
            parts = s.split(',')
            val = float(s.replace(',', '')) if all(len(p) == 3 for p in parts[1:]) \
                else float(s.replace(',', '.'))
        elif '.' in s:
            parts = s.split('.')
            if len(parts) == 2 and len(parts[-1]) == 2:
                val = float(s)                      # 173.00
            elif all(len(p) == 3 for p in parts[1:]) and not has_k:
                val = float(s.replace('.', ''))     # 13.377 -> 13377
            else:
                val = float(s)                      # 0.327 (with k) / 3.189 (amb.)
        else:
            val = float(s)
    except ValueError:
        return np.nan
    if has_k:
        val *= 1000.0
    # '3.189' with no k and 3 decimals already handled as 3189
    return val


def extract_price_from_text(txt: str) -> float:
    for pat in PATTERNS:
        m = pat.search(txt)
        if m:
            v = parse_num(m.group(1), m.group(2) if (m.lastindex or 0) >= 2 else None)
            if v and v > 0:
                return v
    for pat in WORD_PATTERNS:
        m = pat.search(txt)
        if m:
            v = words_to_num(m.group(1))
            if v and v > 0:
                return v
    return np.nan


def extract_price(row) -> float:
    for col in ('lattest comment', 'description', 'neighborhood_overview', 'name',
                'host_about'):
        txt = row.get(col)
        if pd.isna(txt):
            continue
        v = extract_price_from_text(str(txt))
        if not np.isnan(v):
            return v
    return np.nan


if __name__ == '__main__':
    tr = pd.read_csv('train.csv')
    te = pd.read_csv('test.csv')
    for name, df in [('train', tr), ('test', te)]:
        df['leak_price'] = df.apply(extract_price, axis=1)
        print(f'{name}: coverage = {df["leak_price"].notna().mean():.4f}')
    print('\nper-target medians:')
    print(tr.dropna(subset=['leak_price']).groupby('target')['leak_price']
          .agg(['count', 'median']).round(0))
    tr[['id', 'leak_price']].to_csv('leak_train.csv', index=False)
    te[['id', 'leak_price']].to_csv('leak_test.csv', index=False)
    print('saved leak_train.csv / leak_test.csv')
