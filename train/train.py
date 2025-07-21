import pandas as pd
import numpy as np
import re
from collections import defaultdict
from datetime import datetime

class FraudRiskScorer:
    def __init__(self):
        self.SUSPICIOUS_PATTERNS = {
            r'\b(fake|test|sample)\b': 0.5,
        }
        self.PATTERN_MULTIPLIER = 3

        self.RESIDENTIAL_INDICATORS = {
            'apt': 0.8, 'unit': 0.7, '#': 0.7, 'suite': 0.4, 'floor': 0.3,
            'po box': 0.9, 'p.o.': 0.9, 'box': 0.8, 'residence': 0.9,
            'residential': 0.9, 'apartment': 0.9, 'house': 0.8, 'condo': 0.8,
            'room': 0.9
        }
        self.COMMERCIAL_INDICATORS = {
            'plaza': -0.7, 'building': -0.5, 'tower': -0.6, 'office': -0.7,
            'complex': -0.5, 'center': -0.5, 'mall': -0.8, 'commercial': -0.8,
            'industrial': -0.8, 'park': -0.4, 'warehouse': -0.8, 'factory': -0.8,
            'store': -0.7, 'shop': -0.6
        }
        self.RES_THRESHOLD = 0.7
        self.RES_POINTS = 10

        self.MULTI_BUS_RES_POINTS = 15
        self.MULTI_BUS_COMM_POINTS = 8

        self.GEO_CLUSTER_COUNT = 2
        self.GEO_CLUSTER_POINTS = 18


        self.EMP_RATIO_THRESH = 4000
        self.EMP_RATIO_POINTS = 15
        self.VHIGH_RATIO_POINTS = 15

        self.ONE_EMP_POINTS = 30
        self.TWO_EMP_POINTS = 20

        self.PROG_MAX_AMOUNT = 20000
        self.PROG_MAX_POINTS = 25

        self.BATCH_UNIFORM_MIN = 5
        self.BATCH_UNIFORM_PCT = 0.10
        self.BATCH_UNIFORM_POINTS = 15

        self.ZIP_SPIKE_MIN = 5
        self.ZIP_SPIKE_MULT = 5
        self.ZIP_SPIKE_POINTS = 20

        self.SBA_SPIKE_MULT = 15
        self.SBA_SPIKE_POINTS = 8

        self.DEMO_MISSING_POINTS = 10

        self.SEQ_POINTS = 25 

        self._compile_patterns()

    def _compile_patterns(self):
        self._name_pattern_regex = []
        for pat, w in self.SUSPICIOUS_PATTERNS.items():
            self._name_pattern_regex.append((re.compile(pat, re.IGNORECASE), w))

    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['RiskScore'] = 0.0

        df['__addr'] = df['BorrowerAddress'].str.lower().fillna('')
        df['__name'] = df['BorrowerName'].str.lower().fillna('')
        df['__zip5']  = df['BorrowerZip'].astype(str).str[:5]
        df['__latr']  = df['Latitude'].round(3)
        df['__lonr']  = df['Longitude'].round(3)
        df['__amt']   = df['InitialApprovalAmount'].astype(float)
        df['__jobs']  = df['JobsReported'].fillna(0).astype(float)

        def _score_name(name):
            s = 0
            for rx,w in self._name_pattern_regex:
                if rx.search(name):
                    s += self.PATTERN_MULTIPLIER * w
            return s
        df['RiskScore'] += df['__name'].map(_score_name)

        def _res_sum(addr):
            s = 0
            for ind, w in self.RESIDENTIAL_INDICATORS.items():
                if ind in addr: s += w
            for ind, w in self.COMMERCIAL_INDICATORS.items():
                if ind in addr: s += w
            return s
        df['__res_sum'] = df['__addr'].map(_res_sum)
        df.loc[df['__res_sum'] > self.RES_THRESHOLD, 'RiskScore'] += self.RES_POINTS

        addr_counts = df.groupby('__addr').size()
        df['__addr_count'] = df['__addr'].map(addr_counts)
        mask_multi = df['__addr_count'] > 1
        df.loc[mask_multi & (df['__res_sum'] > self.RES_THRESHOLD), 
               'RiskScore'] += (df.loc[mask_multi, '__addr_count'] - 1) * self.MULTI_BUS_RES_POINTS
        df.loc[mask_multi & (df['__res_sum'] <= self.RES_THRESHOLD), 
               'RiskScore'] += (df.loc[mask_multi, '__addr_count'] - 1) * self.MULTI_BUS_COMM_POINTS

        geo_counts = df.groupby(['__latr','__lonr']).size()
        df['__geo_count'] = df.set_index(['__latr','__lonr']).index.map(geo_counts)
        df.loc[df['__geo_count'] > self.GEO_CLUSTER_COUNT, 'RiskScore'] += self.GEO_CLUSTER_POINTS

        df['__ratio'] = df['__amt'] / df['__jobs'].replace(0, np.nan)
        mask_ratio = df['__ratio'] > self.EMP_RATIO_THRESH
        df.loc[mask_ratio, 'RiskScore'] += self.EMP_RATIO_POINTS + self.VHIGH_RATIO_POINTS

        df.loc[df['__jobs'] == 1, 'RiskScore'] += self.ONE_EMP_POINTS
        df.loc[df['__jobs'] == 2, 'RiskScore'] += self.TWO_EMP_POINTS

        df.loc[df['__amt'] == self.PROG_MAX_AMOUNT, 'RiskScore'] += self.PROG_MAX_POINTS

        seq_bonus = defaultdict(bool)
        for lender, sub in df.groupby('OriginatingLender'):
            nums = sorted(int(m.group()) for m in sub['LoanNumber'].str.extract(r'(\d+)$', expand=False).dropna().astype(int))
            if any(b - a == 1 for a, b in zip(nums, nums[1:])):
                seq_bonus[lender] = True
        df.loc[df['OriginatingLender'].map(seq_bonus), 'RiskScore'] += self.SEQ_POINTS

        for (lender, date, zip5), sub in df.groupby(['OriginatingLender','DateApproved','__zip5']):
            if len(sub) >= self.BATCH_UNIFORM_MIN:
                mn, mx = sub['__amt'].min(), sub['__amt'].max()
                if (mx - mn) <= self.BATCH_UNIFORM_PCT * mn:
                    df.loc[sub.index, 'RiskScore'] += self.BATCH_UNIFORM_POINTS

        zip_day = df.groupby(['DateApproved','__zip5']).size().rename('count')
        avg_zip = zip_day.groupby(level=1).mean()
        for (date, zip5), c in zip_day.items():
            if c > self.ZIP_SPIKE_MIN and (c / avg_zip[zip5]) > self.ZIP_SPIKE_MULT:
                df.loc[(df['DateApproved']==date)&(df['__zip5']==zip5), 'RiskScore'] += self.ZIP_SPIKE_POINTS

        office_day = df.groupby(['DateApproved','SBAOfficeCode']).size().rename('count')
        avg_off = office_day.groupby(level=1).mean()
        for (date, ofc), c in office_day.items():
            if ofc and (c / avg_off[ofc]) > self.SBA_SPIKE_MULT:
                df.loc[(df['DateApproved']==date)&(df['SBAOfficeCode']==ofc), 'RiskScore'] += self.SBA_SPIKE_POINTS

        demo = df[['Race','Gender','Ethnicity']].apply(lambda row: row.str.lower().isin(['unanswered']).all(), axis=1)
        df.loc[demo, 'RiskScore'] += self.DEMO_MISSING_POINTS

        return df[['LoanNumber','BorrowerName','RiskScore']]

def calculate_fraud_score(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv, dtype=str)
    scorer = FraudRiskScorer()
    out = scorer.calculate_scores(df)
    out.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python fraud_risk_scorer.py <input.csv> <output.csv>")
    else:
        calculate_fraud_score(sys.argv[1], sys.argv[2])
