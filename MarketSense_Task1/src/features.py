"""Feature engineering for MarketSense Task 1.

Shared between local experiments and the Kaggle notebook.
"""
import re
import ast
import numpy as np
import pandas as pd


# ---------- Normalizers for intentionally-noisy categorical columns ----------

def normalize_city(s: str) -> str:
    """'CITY-C', 'cty_c', 'ID_City_B_01', 'City_C_M3tropolitan' -> 'c', 'b', ..."""
    s = str(s).lower().replace('3', 'e')
    m = re.search(r'(?:city|cty)[^a-z]*([abcd])(?![a-z])', s)
    return m.group(1) if m else 'unk'


def normalize_room_type(s: str) -> str:
    s = str(s).lower().replace('3', 'e')
    s = re.sub(r'[^a-z]', ' ', s)
    if any(k in s for k in ('entire', 'entyre', 'entir', 'whole', 'full house',
                            'apartemen', 'residential')):
        return 'entire'
    if any(k in s for k in ('privat', 'privte', 'own room')):
        return 'private'
    if any(k in s for k in ('shar', 'shre', 'kamar')):
        return 'shared'
    if any(k in s for k in ('hotel', 'hotell', 'hoteel', 'htl', 'boutique')):
        return 'hotel'
    return 'other'


def parse_percent(s) -> float:
    if pd.isna(s):
        return np.nan
    m = re.search(r'(\d+)', str(s))
    return float(m.group(1)) if m else np.nan


def parse_bathrooms_text(s):
    """'1.5 shared baths' -> (1.5, shared=1, private=0). 'Half-bath' -> 0.5."""
    if pd.isna(s):
        return np.nan, 0, 0
    s = str(s).lower()
    shared = int('shared' in s)
    private = int('private' in s)
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    n = float(m.group(1)) if m else (0.5 if 'half' in s else np.nan)
    return n, shared, private


KEY_AMENITIES = [
    'wifi', 'kitchen', 'air conditioning', 'pool', 'free parking', 'gym',
    'elevator', 'washer', 'dryer', 'dishwasher', 'bathtub', 'balcony',
    'heating', 'tv', 'coffee', 'hot tub', 'bbq', 'crib', 'workspace',
    'self check-in', 'lockbox', 'breakfast', 'long term stays', 'pets allowed',
]


def count_words(s) -> int:
    return 0 if pd.isna(s) else len(str(s).split())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready features. Pure per-row transforms (no target usage)."""
    out = pd.DataFrame(index=df.index)

    # ---- normalized categoricals ----
    out['city'] = df['city'].map(normalize_city)
    out['room_type'] = df['room_type'].map(normalize_room_type)
    out['property_type'] = (df['property_type'].astype(str).str.lower()
                            .str.replace(r'[^a-z ]', ' ', regex=True)
                            .str.replace(r'\s+', ' ', regex=True).str.strip())
    out['neighbourhood'] = df['neighbourhood'].fillna('missing')
    out['host_response_time'] = df['host_response_time'].fillna('missing')

    # ---- numeric passthrough ----
    num_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds',
                'minimum_nights', 'maximum_nights',
                'availability_30', 'availability_60', 'availability_90',
                'availability_365', 'number_of_reviews',
                'estimated_occupancy_l365d', 'review_scores_rating',
                'review_scores_accuracy', 'review_scores_cleanliness',
                'review_scores_communication', 'reviews_per_month',
                'host_total_listings_count', 'latitude', 'longitude']
    for c in num_cols:
        out[c] = pd.to_numeric(df[c], errors='coerce')

    # ---- percentages ----
    out['host_response_rate'] = df['host_response_rate'].map(parse_percent)
    out['host_acceptance_rate'] = df['host_acceptance_rate'].map(parse_percent)

    # ---- booleans ----
    for c in ['host_has_profile_pic', 'host_identity_verified',
              'has_availability', 'instant_bookable']:
        out[c] = (df[c] == 't').astype(int)
    out['has_license'] = df['license'].notna().astype(int)

    # ---- bathrooms_text ----
    bt = df['bathrooms_text'].map(parse_bathrooms_text)
    out['bath_n'] = [t[0] for t in bt]
    out['bath_shared'] = [t[1] for t in bt]
    out['bath_private'] = [t[2] for t in bt]
    out['bathrooms'] = out['bathrooms'].fillna(out['bath_n'])

    # ---- dates ----
    ref = pd.to_datetime(df['date_obtained'], errors='coerce')
    for c in ['host_since', 'first_review', 'last_review']:
        d = pd.to_datetime(df[c], errors='coerce')
        out[f'days_{c}'] = (ref - d).dt.days

    # ---- amenities ----
    def parse_amen(s):
        try:
            return [a.lower() for a in ast.literal_eval(str(s))]
        except (ValueError, SyntaxError):
            return []
    amen = df['amenities'].map(parse_amen)
    out['n_amenities'] = amen.map(len)
    for k in KEY_AMENITIES:
        out[f'am_{k.replace(" ", "_")}'] = amen.map(
            lambda lst, k=k: int(any(k in a for a in lst)))

    # ---- host_verifications ----
    hv = df['host_verifications'].fillna('[]').astype(str)
    out['n_verifications'] = hv.str.count(',') + hv.str.contains(r'\w').astype(int)
    out['verif_email'] = hv.str.contains('email').astype(int)
    out['verif_phone'] = hv.str.contains('phone').astype(int)
    out['verif_work_email'] = hv.str.contains('work_email').astype(int)

    # ---- text lengths ----
    for c in ['name', 'description', 'neighborhood_overview', 'host_about',
              'lattest comment']:
        out[f'len_{c.replace(" ", "_")}'] = df[c].map(count_words)

    # ---- interactions / ratios ----
    out['beds_per_person'] = out['beds'] / out['accommodates'].clip(lower=1)
    out['baths_per_person'] = out['bathrooms'] / out['accommodates'].clip(lower=1)
    out['beds_per_bedroom'] = out['beds'] / out['bedrooms'].clip(lower=1)
    out['occ_x_reviews'] = (out['estimated_occupancy_l365d']
                            * np.log1p(out['number_of_reviews']))
    out['avail_ratio_30_365'] = out['availability_30'] / (out['availability_365'] + 1)
    out['review_span_days'] = out['days_first_review'] - out['days_last_review']
    out['reviews_per_day_active'] = (out['number_of_reviews']
                                     / out['review_span_days'].clip(lower=1))
    out['min_nights_log'] = np.log1p(out['minimum_nights'])
    out['max_nights_log'] = np.log1p(out['maximum_nights'].clip(upper=10000))

    # ---- location cluster id (only ~845 unique coordinates in data) ----
    out['latlon_key'] = (out['latitude'].round(3).astype(str) + '_'
                         + out['longitude'].round(3).astype(str))

    return out


def add_group_features(all_df: pd.DataFrame, raw_all: pd.DataFrame) -> pd.DataFrame:
    """Frequency / aggregate features computed over train+test combined
    (no target involved, so no leakage)."""
    out = all_df.copy()

    # host-level aggregates
    host = raw_all.groupby('host_id').agg(
        host_n_listings=('id', 'count'),
        host_mean_reviews=('number_of_reviews', 'mean'),
        host_mean_occ=('estimated_occupancy_l365d', 'mean'),
        host_mean_accom=('accommodates', 'mean'),
    )
    out = out.join(host, on=raw_all['host_id'])
    out['host_id_freq'] = out['host_n_listings']

    # location-cluster aggregates
    grp = out.groupby('latlon_key')
    out['loc_count'] = grp['accommodates'].transform('count')
    out['loc_mean_accom'] = grp['accommodates'].transform('mean')
    out['loc_mean_occ'] = grp['estimated_occupancy_l365d'].transform('mean')
    out['loc_mean_reviews'] = grp['number_of_reviews'].transform('mean')

    # neighbourhood frequency
    out['neigh_freq'] = out.groupby('neighbourhood')['accommodates'].transform('count')
    out['prop_freq'] = out.groupby('property_type')['accommodates'].transform('count')

    return out
