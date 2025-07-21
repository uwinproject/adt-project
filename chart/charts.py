import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df_full = pd.read_csv('full.csv', parse_dates=['DateApproved', 'LoanStatusDate', 'ForgivenessDate'])
df_risk = pd.read_csv('score.csv')
df_risk = df_risk.iloc[:, :3]

df_flagged = df_full.merge(df_risk[['LoanNumber','RiskScore']], on='LoanNumber', how='inner')

# Chart 1
plt.figure(figsize=(8,5))
plt.hist(df_risk['RiskScore'], bins=30, edgecolor='black')
plt.title('Distribution of Fraud Risk Scores (>140)')
plt.xlabel('Risk Score')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Chart 2
state_counts = df_flagged['BorrowerState'].value_counts().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
state_counts.plot(kind='bar')
plt.title('Top 10 States by # of High-Risk PPP Loans')
plt.xlabel('State')
plt.ylabel('Number of Loans')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 3
plt.figure(figsize=(8,6))
plt.scatter(df_flagged['InitialApprovalAmount'], df_flagged['RiskScore'], alpha=0.6)
plt.title('Risk Score vs. Initial Approval Amount')
plt.xlabel('Initial Approval Amount ($)')
plt.ylabel('Risk Score')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Chart 4
df_flagged['Month'] = df_flagged['DateApproved'].dt.to_period('M').dt.to_timestamp()
monthly = df_flagged.groupby('Month').size()
plt.figure(figsize=(10,5))
plt.plot(monthly.index, monthly.values, marker='o')
plt.title('Monthly Trend of High-Risk PPP Loan Approvals')
plt.xlabel('Approval Month')
plt.ylabel('Number of Loans')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Chart 5
df_flagged['Month'] = df_flagged['DateApproved'].dt.to_period('M').dt.to_timestamp()
grouped = df_flagged.groupby('Month')['RiskScore']
monthly_stats = pd.DataFrame({
    'median' : grouped.median(),
    '75th'   : grouped.quantile(0.75),
    '90th'   : grouped.quantile(0.90),
})

plt.figure(figsize=(12,6))
plt.plot(monthly_stats.index, monthly_stats['median'], marker='o', label='50th (median)')
plt.plot(monthly_stats.index, monthly_stats['75th'], marker='s', label='75th percentile')
plt.plot(monthly_stats.index, monthly_stats['90th'], marker='^', label='90th percentile')

peak_month = monthly_stats['90th'].idxmax()
peak_value = monthly_stats.loc[peak_month, '90th']
plt.annotate(
    f'Peak: {peak_value:.0f}',
    xy=(peak_month, peak_value),
    xytext=(peak_month, peak_value + 10),
    arrowprops=dict(arrowstyle='->', lw=1.5),
    ha='center'
)

plt.title('Monthly Fraud‐Risk Score Distribution (50th, 75th, 90th Percentiles)')
plt.xlabel('Approval Month')
plt.ylabel('Risk Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Chart 6 Heatmap
df_flagged = (
    df_risk
    .merge(df_full[['LoanNumber','BorrowerZip']], on='LoanNumber', how='left')
    .assign(BorrowerZip=lambda d: d['BorrowerZip'].str.zfill(5))
)

zip_counts = df_flagged['BorrowerZip'].value_counts().nlargest(50)
zips, counts = zip_counts.index.tolist(), zip_counts.values

cols = 5
rows = int(np.ceil(len(counts) / cols))  
grid = np.zeros((rows, cols), int)
for i, v in enumerate(counts):
    r, c = divmod(i, cols)
    grid[r, c] = v

fig, ax = plt.subplots(figsize=(5, 12), dpi=100) 
im = ax.imshow(grid, cmap='viridis', aspect='equal')
ax.set_aspect('equal') 

for i in range(len(counts)):
    r, c = divmod(i, cols)
    ax.text(
        c, r,
        f"{zips[i]}\n{grid[r, c]}",
        ha='center', va='center',
        color='white',
        fontsize=6
    )

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Top 50 ZIP Codes by High-Risk PPP Loan Count', pad=12)

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Number of High-Risk Loans')

plt.tight_layout()
plt.show()
