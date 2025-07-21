import pandas as pd
import numpy as np
from difflib import SequenceMatcher

GENERIC_TERMS = ['LLC', 'INC', 'COMPANY', 'CORP', 'SERVICES', 'BUSINESS']
GENERIC_TERM_POINTS = 5
SIMILARITY_THRESHOLD = 0.9
SIMILAR_NAME_POINTS = 10

ADDRESS_INDICATORS = {
    'APT': 5,
    'PO BOX': 10
}
ADDRESS_INDICATOR_THRESHOLD = 15
ADDRESS_THRESHOLD_BONUS = 10
MULTI_BUSINESS_ADDRESS_POINTS = 15 

EMPLOYEE_THRESHOLDS = [
    (14_000, 15),
    (11_000, 15),
]
EMPLOYEE_COUNT_POINTS = {
    1: 30,
    2: 20
}

MISSING_DEMO_POINTS = 10
HIGH_RISK_BUSINESS_TYPES = {
    'CANNABIS': 20,
    'GAMBLING': 25,
    'FIREARMS': 25,
    'ENTERTAINMENT': 30
}

SEQUENTIAL_LOAN_POINTS = 10
ZIP_SPIKE_MULTIPLIER = 2.0
ZIP_SPIKE_POINTS = 12.5
OFFICE_SPIKE_MULTIPLIER = 2.5
OFFICE_SPIKE_POINTS = 12.5


class FraudRiskScorer:
    def __init__(self):
        pass

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['risk_score'] = 0

        self._score_name_terms(df)
        self._score_name_similarity(df)

        self._score_address_indicators(df)
        self._score_multi_business_address(df)

        self._score_amount_per_employee(df)

        self._score_missing_demographics(df)
        self._score_high_risk_business_type(df)

        self._score_sequential_loans(df)
        self._score_zip_spikes(df)
        self._score_office_spikes(df)

        return df

    def _score_name_terms(self, df):
        def generic_score(name):
            score = 0
            upper = name.upper()
            for term, pts in zip(GENERIC_TERMS, [GENERIC_TERM_POINTS]*len(GENERIC_TERMS)):
                if term in upper:
                    score += pts
            return score

        df['risk_score'] += df['business_name'].fillna('').apply(generic_score)

    def _score_name_similarity(self, df):
        names = df['business_name'].fillna('').tolist()
        for i, name_i in enumerate(names):
            for j in range(i+1, len(names)):
                name_j = names[j]
                if not name_i or not name_j:
                    continue
                sim = SequenceMatcher(None, name_i.lower(), name_j.lower()).ratio()
                if sim >= SIMILARITY_THRESHOLD:
                    df.at[i, 'risk_score'] += SIMILAR_NAME_POINTS
                    df.at[j, 'risk_score'] += SIMILAR_NAME_POINTS

    def _score_address_indicators(self, df):
        def addr_indicator_score(addr):
            score = 0
            upper = addr.upper()
            for indicator, pts in ADDRESS_INDICATORS.items():
                if indicator in upper:
                    score += pts
            return score

        df['__addr_pts'] = df['address'].fillna('').apply(addr_indicator_score)
        df['risk_score'] += df['__addr_pts']
        df.loc[df['__addr_pts'] > ADDRESS_INDICATOR_THRESHOLD, 'risk_score'] += ADDRESS_THRESHOLD_BONUS
        df.drop(columns='__addr_pts', inplace=True)

    def _score_multi_business_address(self, df):
        addr_counts = df.groupby('address')['loan_id'].transform('count')
        extra = (addr_counts - 1).clip(lower=0)
        df['risk_score'] += extra * MULTI_BUSINESS_ADDRESS_POINTS

    def _score_amount_per_employee(self, df):
        amt = df['loan_amount']
        emp = df['employees'].replace(0, np.nan)
        ratio = amt / emp
        for thresh, pts in EMPLOYEE_THRESHOLDS:
            df.loc[ratio > thresh, 'risk_score'] += pts
        for cnt, pts in EMPLOYEE_COUNT_POINTS.items():
            df.loc[df['employees'] == cnt, 'risk_score'] += pts

    def _score_missing_demographics(self, df):
        demo_cols = ['gender', 'race', 'ethnicity']
        is_missing_all = df[demo_cols].isnull().all(axis=1)
        df.loc[is_missing_all, 'risk_score'] += MISSING_DEMO_POINTS

    def _score_high_risk_business_type(self, df):
        upper_types = df['business_type'].str.upper().fillna('')
        for btype, pts in HIGH_RISK_BUSINESS_TYPES.items():
            df.loc[upper_types == btype, 'risk_score'] += pts

    def _score_sequential_loans(self, df):
        df = df.sort_values(['business_name', 'loan_number'])
        prev_num = df.groupby('business_name')['loan_number'].shift(1)
        df['__seq'] = (df['loan_number'].astype(int) - prev_num.astype(float) == 1)
        df.loc[df['__seq'], 'risk_score'] += SEQUENTIAL_LOAN_POINTS
        df.drop(columns='__seq', inplace=True)

    def _score_zip_spikes(self, df):
        grp = df.groupby(['zip_code', 'date'])['loan_id'].transform('count')
        avg = df.groupby('zip_code')['loan_id'].transform(lambda x: x.count() / x.index.to_series().map(lambda i: df.at[i, 'date']).nunique())
        spike = grp > (avg * ZIP_SPIKE_MULTIPLIER)
        df.loc[spike, 'risk_score'] += ZIP_SPIKE_POINTS

    def _score_office_spikes(self, df):
        grp = df.groupby(['sba_office', 'date'])['loan_id'].transform('count')
        avg = df.groupby('sba_office')['loan_id'].transform(lambda x: x.count() / x.index.to_series().map(lambda i: df.at[i, 'date']).nunique())
        spike = grp > (avg * OFFICE_SPIKE_MULTIPLIER)
        df.loc[spike, 'risk_score'] += OFFICE_SPIKE_POINTS

if __name__ == "__main__":
    ppd = pd.read_csv("ppp_loans.csv", parse_dates=['date'])
  
    scorer = FraudRiskScorer()
    scored = scorer.score(ppd)
    
    out = scored.loc[:, ['loan_number', 'business_name', 'risk_score']].copy()
    out.columns = ['LoanNumber', 'BorrowerName', 'RiskScore']
    
    out.to_csv("loan_risk_scores.csv", index=False)
