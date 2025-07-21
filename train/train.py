import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    classification_report
)

import xgboost as xgb
import joblib

df_ppp   = pd.read_csv('filtered-dataset.csv')
df_high  = pd.read_csv('final-output.csv')

df = df_ppp.merge(
    df_high[['LoanNumber']], 
    on='LoanNumber', 
    how='left', 
    indicator=True
)

df['HighRisk'] = (df['_merge'] == 'both').astype(int)
df = df.drop(columns=['_merge'])

df = df.drop(columns=['LoanNumber', 'BorrowerName', 'RiskScore'], errors='ignore')

obj_cols = df.select_dtypes(include=['object', 'category']).columns
df[obj_cols] = df[obj_cols].fillna('Unknown').astype(str)

y = df['HighRisk']
X = df.drop(columns='HighRisk')

# 6. Identify numeric vs categorical features
num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = X.select_dtypes(include=['object']).columns.tolist()

num_pipe = SimpleImputer(strategy='median')
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_feats),
    ('cat', cat_pipe, cat_feats),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

best_params = {
    'n_estimators':       100,
    'learning_rate':      0.1,
    'max_depth':          5,
    'min_child_weight':   3,
    'subsample':          1.0,
    'colsample_bytree':   1.0,
    'scale_pos_weight':   392.4515738498789
}

clf = Pipeline([
    ('pre', preprocessor),
    ('xgb', xgb.XGBClassifier(
        **best_params,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist',
        n_jobs=12,
        max_bin=256
    ))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("Accuracy:          ", accuracy_score(y_test, y_pred))
print("ROC AUC:           ", roc_auc_score(y_test, y_proba))
print("Average Precision: ", average_precision_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(clf, 'xgb_highrisk_classifier.joblib')
