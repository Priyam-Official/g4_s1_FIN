import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt

# Assets and portfolio weights
A = ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMD']
W = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
F = ['XLK', 'MTUM', 'VLUE']
C = 0.95
Z = norm.ppf(C)

def load_and_prepare_data(file_path):
    """Load CSV data and extract returns for assets and factors"""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Extract daily returns for assets
    asset_returns = pd.DataFrame()
    for asset in A:
        asset_returns[asset] = df[f'return_{A.index(asset)+1}']
    
    # Extract factor returns
    factor_returns = pd.DataFrame({
        'XLK': df['returns_dj'],
        'MTUM': df['returns_nasdaq'],
        'VLUE': df['returns_SP500']
    })
    
    return asset_returns, factor_returns

print("Loading data...")
asset_returns, factor_returns = load_and_prepare_data('enhanced_stock_dataset.csv')

# Split into train and test sets
split_idx = int(0.8 * len(asset_returns))
train_dates = asset_returns.index[:split_idx]
test_dates = asset_returns.index[split_idx:]

R_a = asset_returns.loc[train_dates]
R_f = factor_returns.loc[train_dates]
R_a_te = asset_returns.loc[test_dates]
R_f_te = factor_returns.loc[test_dates]

# Portfolio returns
R_p = R_a.dot(W)
R_p_te = R_a_te.dot(W)
L_te = -R_p_te

print(f"Training period: {train_dates[0].date()} to {train_dates[-1].date()}")
print(f"Testing period: {test_dates[0].date()} to {test_dates[-1].date()}")
print(f"Total samples: {len(asset_returns)}")
print(f"Training samples: {len(R_p)}")
print(f"Testing samples: {len(R_p_te)}")

# Linear Factor Model
X = sm.add_constant(R_f)
M = sm.OLS(R_p, X).fit()
B = M.params.drop('const')
alpha = M.params['const']
residuals = M.resid
sigma_e = np.std(residuals)

print("\n" + "="*50)
print("MODEL FIT STATISTICS")
print("="*50)
print(f"R-squared: {M.rsquared:.4f}")
print(f"Adj. R-squared: {M.rsquared_adj:.4f}")
print(f"Idiosyncratic volatility: {sigma_e:.4f}")
print("\nFactor Sensitivities:")
for factor, beta in B.items():
    print(f"  {factor}: {beta:.4f}")
print(f"Alpha: {alpha:.4f}")

# Factor moments
m_f = R_f.mean().values
V_f = R_f.cov().values

# Portfolio return distribution
mu_R = np.dot(B.values, m_f)
var_R_factor = np.dot(np.dot(B.values, V_f), B.values.T)
var_R_total = var_R_factor + sigma_e**2
sigma_R = np.sqrt(var_R_total)

# Loss distribution
mu_L = -mu_R
sigma_L = sigma_R

# Closed-form VaR
VaR_95 = mu_L + Z * sigma_L
VaR_99 = mu_L + norm.ppf(0.99) * sigma_L

print("\n" + "="*50)
print("GAUSSIAN MODEL RESULTS")
print("="*50)
print(f"Portfolio Expected Return: {mu_R:.4f}%")
print(f"Portfolio Volatility: {sigma_R:.4f}%")
print(f"Factor Contribution to Variance: {var_R_factor:.6f}")
print(f"Idiosyncratic Contribution: {sigma_e**2:.6f}")
print(f"Loss Distribution Mean: {mu_L:.4f}%")
print(f"Loss Distribution Std Dev: {sigma_L:.4f}%")
print(f"95% VaR: {VaR_95:.4f}%")
print(f"99% VaR: {VaR_99:.4f}%")

#Loss Distributions with improved spacing
x = np.linspace(mu_L - 4*sigma_L, mu_L + 4*sigma_L, 1000)
pdf_loss = norm.pdf(x, mu_L, sigma_L)
cdf_loss = norm.cdf(x, mu_L, sigma_L)
z_scores = (x - mu_L) / sigma_L
pdf_std = norm.pdf(z_scores)

plt.figure(figsize=(16, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.1)

# Plot 1: Loss PDF with VaR
plt.subplot(2, 2, 1)
plt.plot(x, pdf_loss, 'b-', linewidth=2)
plt.axvline(VaR_95, color='red', linestyle='--', linewidth=2, label=f'95% VaR = {VaR_95:.4f}%')
plt.fill_between(x[x <= VaR_95], pdf_loss[x <= VaR_95], alpha=0.3, color='red')
plt.xlabel('Loss (%)', fontsize=11)
plt.ylabel('Probability Density', fontsize=11)
plt.title('Loss Distribution PDF', fontsize=12, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.95, 'Shows probability density of portfolio losses.\nRed area = 5% tail, vertical line = 95% VaR',
         transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Loss CDF with VaR
plt.subplot(2, 2, 2)
plt.plot(x, cdf_loss, 'g-', linewidth=2)
plt.axvline(VaR_95, color='red', linestyle='--', linewidth=2)
plt.axhline(C, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Loss (%)', fontsize=11)
plt.ylabel('Cumulative Probability', fontsize=11)
plt.title('Loss Distribution CDF', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.95, 'Cumulative probability up to each loss level.\nIntersection shows P(Loss ≤ VaR) = 95%',
         transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Standardized Normal
plt.subplot(2, 2, 3)
plt.plot(z_scores, pdf_std, 'b-', linewidth=2)
plt.axvline(Z, color='red', linestyle='--', linewidth=2, label=f'z = {Z:.2f}')
plt.fill_between(z_scores[z_scores <= Z], pdf_std[z_scores <= Z], alpha=0.3, color='red')
plt.xlabel('Standardized Loss: (L - μ_L)/σ_L', fontsize=11)
plt.ylabel('Probability Density', fontsize=11)
plt.title('Standardized Normal Distribution', fontsize=12, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.95, 'Standardized losses follow N(0,1).\nVaR threshold = z = 1.645 (95th percentile)',
         transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Risk Decomposition
plt.subplot(2, 2, 4)
labels = ['Factor Risk', 'Idiosyncratic Risk']
sizes = [var_R_factor, sigma_e**2]
colors = ['#ff9999', '#66b3ff']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
        textprops={'fontsize': 11})
plt.axis('equal')
plt.title('Risk Decomposition', fontsize=12, fontweight='bold')
plt.text(0.5, -0.15, 'Breakdown of total portfolio variance into\nsystematic risk (factors) and asset-specific risk',
         transform=plt.gca().transAxes, fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Deterministic Gaussian VaR Model - Distribution Analysis', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Deterministic Backtesting
print("\n" + "="*50)
print("BACKTESTING RESULTS")
print("="*50)

E = L_te > VaR_95
nE = E.sum()
N = len(L_te)
exc_rate = nE / N
exp_exc = N * (1 - C)

print(f"Test Period: {test_dates[0].date()} to {test_dates[-1].date()}")
print(f"Total Days: {N}")
print(f"Expected Exceptions: {exp_exc:.1f}")
print(f"Actual Exceptions: {nE}")
print(f"Exception Rate: {exc_rate:.4f}")
print(f"Model Status: {'Acceptable' if exc_rate < 0.07 else 'Reject'}")

# Plot backtest 
plt.figure(figsize=(14, 7))
plt.plot(L_te.index, L_te, 'b-', alpha=0.7, linewidth=1, label='Actual Daily Loss')
plt.axhline(VaR_95, color='red', linestyle='--', linewidth=2, label=f'95% VaR ({VaR_95:.4f}%)')

exc_dates = L_te.index[E]
exc_values = L_te[E]
plt.scatter(exc_dates, exc_values, color='red', s=50, marker='x', zorder=5, 
            label=f'Exceptions (n={nE})')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Loss (%)', fontsize=12)
plt.title('Deterministic VaR Backtest - Actual Losses vs VaR Threshold', fontsize=13, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.95, f'Actual daily losses compared to fixed VaR threshold.\nRed X marks = exceptions ({nE} days, {exc_rate:.2%})',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summary Statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print("\nAsset Returns (Annualized):")
annualization_factor = 252
for asset in A:
    mean_return = R_a[asset].mean() * annualization_factor
    volatility = R_a[asset].std() * np.sqrt(annualization_factor)
    print(f"  {asset}: Mean={mean_return:.2%}, Vol={volatility:.2%}")

print("\nFactor Correlations:")
print(R_f.corr().round(3))

print(f"\nPortfolio Statistics:")
print(f"  Expected Annual Return: {mu_R * annualization_factor:.2%}")
print(f"  Annual Volatility: {sigma_R * np.sqrt(annualization_factor):.2%}")
print(f"  Sharpe Ratio: {(mu_R * annualization_factor) / (sigma_R * np.sqrt(annualization_factor)):.2f}")
print(f"  95% Annual VaR: {VaR_95 * np.sqrt(annualization_factor):.2%}")