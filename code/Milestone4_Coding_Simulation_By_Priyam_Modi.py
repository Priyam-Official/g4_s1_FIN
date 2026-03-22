"""
MILESTONE 4: RANDOMIZED ALGORITHM FOR VAR ESTIMATION
Author: Priyam Modi (AU2440030)
Course: Financial Risk Management / Market Risk Modeling
Comparison: Deterministic vs Monte Carlo Simulation
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, chi2
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
import time
from statsmodels.graphics.tsaplots import plot_acf
warnings.filterwarnings('ignore')

# Configuration
A = ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMD']
W = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
F = ['XLK', 'MTUM', 'VLUE']
C = 0.95
Z = norm.ppf(C)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

COLORS = {
    'deterministic': '#FF6B6B',
    'montecarlo': '#4ECDC4',
    'actual': '#2C3E50',
    'normal': '#95A5A6'
}

def load_and_prepare_data(file_path):
    """Load CSV data and extract returns"""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    asset_returns = pd.DataFrame()
    for asset in A:
        asset_returns[asset] = df[f'return_{A.index(asset)+1}']
    
    factor_returns = pd.DataFrame({
        'XLK': df['returns_dj'],
        'MTUM': df['returns_nasdaq'],
        'VLUE': df['returns_SP500']
    })
    
    return asset_returns, factor_returns

# Load data
asset_returns, factor_returns = load_and_prepare_data('enhanced_stock_dataset.csv')

# Split data
split_idx = int(0.8 * len(asset_returns))
train_dates = asset_returns.index[:split_idx]
test_dates = asset_returns.index[split_idx:]

R_a = asset_returns.loc[train_dates]
R_f = factor_returns.loc[train_dates]
R_a_te = asset_returns.loc[test_dates]
R_f_te = factor_returns.loc[test_dates]

# Portfolio returns and losses
R_p = R_a.dot(W)
R_p_te = R_a_te.dot(W)
L_te = -R_p_te

# Test window info
test_window_size = len(L_te)
expected_exceptions_95 = test_window_size * 0.05

# Deterministic model
X = sm.add_constant(R_f)
M = sm.OLS(R_p, X).fit()
B = M.params.drop('const')
alpha = M.params['const']
residuals = M.resid
sigma_e = np.std(residuals)

m_f = R_f.mean().values
V_f = R_f.cov().values

mu_R = np.dot(B.values, m_f)
var_R_factor = np.dot(np.dot(B.values, V_f), B.values.T)
var_R_total = var_R_factor + sigma_e**2
sigma_R = np.sqrt(var_R_total)

mu_L = -mu_R
sigma_L = sigma_R

det_VaR_95 = mu_L + Z * sigma_L
det_VaR_99 = mu_L + norm.ppf(0.99) * sigma_L

# Monte Carlo Simulation Class
class MonteCarloVaR:
    def __init__(self, portfolio_returns, n_simulations=10000):
        self.portfolio_returns = portfolio_returns
        self.n_simulations = n_simulations
        self.mean_return = np.mean(portfolio_returns)
        self.std_return = np.std(portfolio_returns)
        self.historical_losses = -portfolio_returns[portfolio_returns < 0]
        
    def estimate_var(self, confidence_level=0.95):
        simulated_returns = np.random.normal(self.mean_return, self.std_return, self.n_simulations)
        simulated_losses = -simulated_returns[simulated_returns < 0]
        
        if len(simulated_losses) < 100 and len(self.historical_losses) > 0:
            simulated_losses = np.random.choice(self.historical_losses, self.n_simulations, replace=True)
        elif len(simulated_losses) < 100:
            simulated_losses = np.random.exponential(0.02, self.n_simulations)
        
        var_estimate = np.percentile(simulated_losses, (1 - confidence_level) * 100)
        return var_estimate, simulated_losses

# Backtesting function
def calculate_backtest_metrics(actual_losses, var_estimate, confidence_level=0.95):
    n = len(actual_losses)
    exceptions = actual_losses > var_estimate
    n_exceptions = np.sum(exceptions)
    expected_exceptions = n * (1 - confidence_level)
    exception_rate = n_exceptions / n
    
    if 0 < n_exceptions < n:
        LR_pof = 2 * (n_exceptions * np.log(n_exceptions / (n * (1 - confidence_level))) + (n - n_exceptions) * np.log((n - n_exceptions) / (n * confidence_level)))
        p_value_pof = 1 - stats.chi2.cdf(LR_pof, 1)
    else:
        p_value_pof = np.nan
    
    yellow_threshold = expected_exceptions * 1.2
    
    if n_exceptions <= expected_exceptions:
        zone = "GREEN"
        zone_color = 'green'
    elif n_exceptions <= yellow_threshold:
        zone = "YELLOW"
        zone_color = 'yellow'
    else:
        zone = "RED"
        zone_color = 'red'
    
    return {
        'n_exceptions': n_exceptions,
        'expected_exceptions': expected_exceptions,
        'exception_rate': exception_rate,
        'p_value_pof': p_value_pof,
        'zone': zone,
        'zone_color': zone_color,
        'exceptions': exceptions
    }

# Run models
mc_model = MonteCarloVaR(R_p, n_simulations=10000)
det_metrics = calculate_backtest_metrics(L_te, det_VaR_95, C)

start_time = time.time()
mc_var, mc_dist = mc_model.estimate_var(C)
mc_time = time.time() - start_time
mc_metrics = calculate_backtest_metrics(L_te, mc_var, C)

# Figure 1: Probability Distribution Analysis
fig1 = plt.figure(figsize=(16, 14))
fig1.suptitle('Figure 1: Probability Distribution Analysis - Deterministic vs Monte Carlo', fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.35, wspace=0.35)

# 1.1 PDF Comparison
ax1 = fig1.add_subplot(gs[0, 0])
x_range = np.linspace(0, 0.10, 1000)
det_pdf = norm.pdf(x_range, mu_L, sigma_L)
ax1.plot(x_range, det_pdf, linewidth=3, label='Deterministic (Normal)', color=COLORS['deterministic'])
ax1.hist(mc_dist, bins=50, density=True, alpha=0.7, label='Monte Carlo', color=COLORS['montecarlo'], edgecolor='black', linewidth=0.5)
ax1.axvline(det_VaR_95, color=COLORS['deterministic'], linestyle='--', linewidth=2.5, label=f'Det VaR: {det_VaR_95:.4f}%')
ax1.axvline(mc_var, color=COLORS['montecarlo'], linestyle='--', linewidth=2.5, label=f'MC VaR: {mc_var:.4f}%')
ax1.set_xlabel('Loss (%)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('PDF Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 0.10])
ax1.set_ylim([0, 60])

# 1.2 CDF Comparison
ax2 = fig1.add_subplot(gs[0, 1])
x_cdf = np.linspace(0, 0.10, 1000)
det_cdf = norm.cdf(x_cdf, mu_L, sigma_L)
ax2.plot(x_cdf, det_cdf, linewidth=3, label='Deterministic', color=COLORS['deterministic'])
mc_ecdf = np.array([np.sum(mc_dist <= x) / len(mc_dist) for x in x_cdf])
ax2.plot(x_cdf, mc_ecdf, linewidth=3, label='Monte Carlo', color=COLORS['montecarlo'])
ax2.axhline(0.95, color='black', linestyle=':', linewidth=2, label='95% Confidence')
ax2.set_xlabel('Loss (%)', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('CDF Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 0.10])
ax2.set_ylim([0, 1])

# 1.3 QQ Plot
ax3 = fig1.add_subplot(gs[1, 0])
mc_dist_pos = mc_dist[mc_dist > 0]
if len(mc_dist_pos) > 10:
    theoretical_q = np.percentile(np.random.normal(0, 1, 10000), np.linspace(0.1, 99.9, 50))
    sample_q = np.percentile((mc_dist_pos - np.mean(mc_dist_pos)) / np.std(mc_dist_pos), np.linspace(0.1, 99.9, 50))
    ax3.scatter(theoretical_q, sample_q, alpha=0.7, s=40, label='Monte Carlo', color=COLORS['montecarlo'], edgecolor='black', linewidth=0.5)
ax3.plot([-4, 4], [-4, 4], 'r--', linewidth=2, label='Normal Reference')
ax3.set_xlabel('Theoretical Normal Quantiles', fontsize=12)
ax3.set_ylabel('Sample Quantiles', fontsize=12)
ax3.set_title('Q-Q Plot: Monte Carlo vs Normal', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-4, 4])
ax3.set_ylim([-4, 4])

# 1.4 VaR Comparison
ax4 = fig1.add_subplot(gs[1, 1])
models = ['Deterministic', 'Monte Carlo']
var_values = [det_VaR_95, mc_var]
colors_bar = [COLORS['deterministic'], COLORS['montecarlo']]
x_pos = np.arange(len(models))
bars = ax4.bar(x_pos, var_values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models, fontsize=11)
ax4.set_ylabel('VaR (95%) - Loss %', fontsize=12)
ax4.set_xlabel('Model', fontsize=12)
ax4.set_title('VaR Estimates Comparison', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, max(var_values) * 1.2])
for i, (bar, val) in enumerate(zip(bars, var_values)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{val:.4f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
plt.subplots_adjust(top=0.93)
plt.show()

# Figure 2: Backtesting Analysis
fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle('Figure 2: Backtesting Analysis - Deterministic vs Monte Carlo', fontsize=16, fontweight='bold', y=0.98)

gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

# 2.1 Deterministic Backtest
ax21 = fig2.add_subplot(gs2[0, 0])
ax21.plot(L_te.index, L_te, linewidth=1.5, label='Actual Loss', color=COLORS['actual'])
ax21.axhline(det_VaR_95, color=COLORS['deterministic'], linestyle='--', linewidth=2.5, label=f'VaR: {det_VaR_95:.4f}%')
exc_dates = L_te.index[det_metrics['exceptions']]
exc_values = L_te[det_metrics['exceptions']]
if len(exc_dates) > 0:
    ax21.scatter(exc_dates, exc_values, color='red', s=60, marker='x', label=f'Exceptions ({det_metrics["n_exceptions"]})', linewidth=2)
ax21.set_xlabel('Date', fontsize=11)
ax21.set_ylabel('Loss (%)', fontsize=11)
ax21.set_title(f'Deterministic Model Backtest - {det_metrics["zone"]} ZONE', fontsize=13, fontweight='bold')
ax21.legend(loc='upper right', fontsize=8)
ax21.grid(True, alpha=0.3)
ax21.tick_params(axis='x', rotation=45)
ax21.set_ylim([0, max(L_te) * 1.1])
ax21.add_patch(Rectangle((0, 0), 1, 1, transform=ax21.transAxes, alpha=0.1, color=det_metrics['zone_color']))

# 2.2 Monte Carlo Backtest
ax22 = fig2.add_subplot(gs2[0, 1])
ax22.plot(L_te.index, L_te, linewidth=1.5, label='Actual Loss', color=COLORS['actual'])
ax22.axhline(mc_var, color=COLORS['montecarlo'], linestyle='--', linewidth=2.5, label=f'VaR: {mc_var:.4f}%')
exc_mc = L_te > mc_var
exc_dates_mc = L_te.index[exc_mc]
exc_values_mc = L_te[exc_mc]
if len(exc_dates_mc) > 0:
    ax22.scatter(exc_dates_mc, exc_values_mc, color=COLORS['montecarlo'], s=60, marker='x', label=f'Exceptions ({len(exc_dates_mc)})', linewidth=2)
ax22.set_xlabel('Date', fontsize=11)
ax22.set_ylabel('Loss (%)', fontsize=11)
ax22.set_title(f'Monte Carlo Backtest - {mc_metrics["zone"]} ZONE', fontsize=13, fontweight='bold')
ax22.legend(loc='upper right', fontsize=8)
ax22.grid(True, alpha=0.3)
ax22.tick_params(axis='x', rotation=45)
ax22.set_ylim([0, max(L_te) * 1.1])
ax22.add_patch(Rectangle((0, 0), 1, 1, transform=ax22.transAxes, alpha=0.1, color=mc_metrics['zone_color']))

# 2.3 Exception Timeline
ax23 = fig2.add_subplot(gs2[1, 0])
exception_timeline = pd.DataFrame({'Date': L_te.index, 'Deterministic': det_metrics['exceptions'].astype(int), 'Monte Carlo': (L_te > mc_var).astype(int)})
exception_timeline.set_index('Date', inplace=True)
cumulative = exception_timeline.cumsum()
for col in cumulative.columns:
    color = COLORS['deterministic'] if col == 'Deterministic' else COLORS['montecarlo']
    ax23.plot(cumulative.index, cumulative[col], linewidth=2.5, label=col, color=color)
expected_line = np.linspace(0, det_metrics['expected_exceptions'], len(cumulative))
ax23.plot(cumulative.index, expected_line, 'k--', linewidth=2, label=f'Expected ({expected_exceptions_95:.1f})', alpha=0.7)
ax23.set_xlabel('Date', fontsize=11)
ax23.set_ylabel('Cumulative Exceptions', fontsize=11)
ax23.set_title('Exception Timeline', fontsize=13, fontweight='bold')
ax23.legend(loc='upper left', fontsize=8)
ax23.grid(True, alpha=0.3)
ax23.tick_params(axis='x', rotation=45)

# 2.4 Statistics Table
ax24 = fig2.add_subplot(gs2[1, 1])
ax24.axis('off')
test_data = [
    ['Deterministic', f"{det_metrics['n_exceptions']} / {det_metrics['expected_exceptions']:.1f}", f"{det_metrics['p_value_pof']:.4f}" if not np.isnan(det_metrics['p_value_pof']) else "N/A", det_metrics['zone']],
    ['Monte Carlo', f"{mc_metrics['n_exceptions']} / {mc_metrics['expected_exceptions']:.1f}", f"{mc_metrics['p_value_pof']:.4f}" if not np.isnan(mc_metrics['p_value_pof']) else "N/A", mc_metrics['zone']],
    ['Test Window', f"{test_window_size} days", f"Expected: {expected_exceptions_95:.1f}", ""]
]
columns = ['Model', 'Exceptions/Expected', 'POF p-value', 'Zone']
table = ax24.table(cellText=test_data, colLabels=columns, cellLoc='center', loc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)
ax24.set_title('Backtesting Statistics', fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.show()

# Figure 3: Risk Decomposition
fig3 = plt.figure(figsize=(16, 10))
fig3.suptitle('Figure 3: Risk Attribution Analysis', fontsize=16, fontweight='bold', y=0.98)

gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.3, wspace=0.3)

# 3.1 Risk Decomposition
ax31 = fig3.add_subplot(gs3[0, 0])
labels = ['Factor Risk', 'Idiosyncratic Risk']
sizes = [var_R_factor, sigma_e**2]
colors_pie = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = ax31.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax31.axis('equal')
ax31.set_title('Portfolio Risk Decomposition', fontsize=13, fontweight='bold')

# 3.2 Factor Correlation
ax32 = fig3.add_subplot(gs3[0, 1])
factor_corr = R_f[['XLK', 'MTUM', 'VLUE']].corr()
im = ax32.imshow(factor_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax32.set_xticks(np.arange(len(factor_corr.columns)))
ax32.set_yticks(np.arange(len(factor_corr.index)))
ax32.set_xticklabels(factor_corr.columns)
ax32.set_yticklabels(factor_corr.index)
ax32.set_title('Factor Correlation Matrix', fontsize=13, fontweight='bold')
for i in range(len(factor_corr.index)):
    for j in range(len(factor_corr.columns)):
        ax32.text(j, i, f'{factor_corr.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontweight='bold', fontsize=10)
plt.colorbar(im, ax=ax32)

# 3.3 Rolling Betas
ax33 = fig3.add_subplot(gs3[0, 2])
window = 252
rolling_betas = {factor: [] for factor in F}
rolling_dates = []
for i in range(window, len(R_p)):
    X_roll = sm.add_constant(R_f.iloc[i-window:i])
    y_roll = R_p.iloc[i-window:i]
    model_roll = sm.OLS(y_roll, X_roll).fit()
    for factor in F:
        rolling_betas[factor].append(model_roll.params[factor])
    rolling_dates.append(R_p.index[i])
for factor in F:
    ax33.plot(rolling_dates, rolling_betas[factor], linewidth=2, label=factor)
ax33.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax33.set_xlabel('Date', fontsize=11)
ax33.set_ylabel('Rolling Beta', fontsize=11)
ax33.set_title('Rolling Factor Sensitivities', fontsize=13, fontweight='bold')
ax33.legend(loc='upper left', fontsize=9)
ax33.tick_params(axis='x', rotation=45)

# 3.4 Residual Plot
ax34 = fig3.add_subplot(gs3[1, 0])
ax34.scatter(M.fittedvalues, M.resid, alpha=0.5, color='#2C3E50', s=30, edgecolor='white', linewidth=0.5)
ax34.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax34.set_xlabel('Fitted Values', fontsize=11)
ax34.set_ylabel('Residuals', fontsize=11)
ax34.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax34.grid(True, alpha=0.3)

# 3.5 Residual Distribution
ax35 = fig3.add_subplot(gs3[1, 1])
ax35.hist(M.resid, bins=40, density=True, alpha=0.7, color='#4ECDC4', edgecolor='black', linewidth=0.5)
x_resid = np.linspace(M.resid.min(), M.resid.max(), 100)
ax35.plot(x_resid, stats.norm.pdf(x_resid, 0, np.std(M.resid)), 'r-', linewidth=2, label='Normal Fit')
ax35.set_xlabel('Residual Value', fontsize=11)
ax35.set_ylabel('Density', fontsize=11)
ax35.set_title('Residual Distribution', fontsize=13, fontweight='bold')
ax35.legend(loc='upper right', fontsize=9)
ax35.grid(True, alpha=0.3)

# 3.6 ACF
ax36 = fig3.add_subplot(gs3[1, 2])
plot_acf(M.resid, ax=ax36, lags=20, alpha=0.05, color='#2C3E50', linewidth=2)
ax36.set_title('Autocorrelation of Residuals', fontsize=13, fontweight='bold')
ax36.set_xlabel('Lag', fontsize=11)
ax36.set_ylabel('Autocorrelation', fontsize=11)
ax36.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()