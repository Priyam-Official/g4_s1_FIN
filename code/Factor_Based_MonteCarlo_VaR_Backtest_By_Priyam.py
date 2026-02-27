import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Configuration & Data Retrieval

# Portfolio Configuration
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMD']
WEIGHTS = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

FACTOR_TICKERS = ['SPY', 'XLK', 'TLT', 'MTUM', 'VLUE']

# Simulation Parameters
CONFIDENCE_LEVEL = 0.95  # 95% VaR (Targeting 5th Percentile)
SIMULATIONS = 10000      # Monte Carlo paths
TRAIN_START = "2023-01-01"
TRAIN_END = "2024-04-30"
TEST_START = "2024-05-01"
TEST_END = "2025-05-01"

def get_data(tickers, start, end):
    """
    Fetch adjusted close prices from Yahoo Finance and compute log returns.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    # Calculate log returns: ln(Pt / Pt-1)
    returns = np.log(data / data.shift(1)).dropna()
    return returns

# Fetch Training (Estimation) and Testing (Backtest) Data
all_tickers = TICKERS + FACTOR_TICKERS
train_data = get_data(all_tickers, TRAIN_START, TRAIN_END)
test_data = get_data(all_tickers, TEST_START, TEST_END)

# Separate Portfolio Assets and Factors
port_train_returns = train_data[TICKERS]
factor_train_returns = train_data[FACTOR_TICKERS]

# Calculate Historical Portfolio Returns (Weighted Sum)
portfolio_train_series = port_train_returns.dot(WEIGHTS)

# 2. Beta Estimation (Factor Regression)
def calculate_vif(X):
    """Check for Multicollinearity using Variance Inflation Factor (VIF)."""
    vif_data = pd.DataFrame()
    vif_data["Factor"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Optional: Check VIF to ensure factors aren't too correlated
vif_df = calculate_vif(factor_train_returns)
print("--- Variance Inflation Factors (VIF) ---")
print(vif_df)
print("\n")

# OLS Regression: Portfolio Returns ~ Factor Returns
# Equation: R_p = alpha + beta_1*F_1 + ... + beta_n*F_n + error
X = sm.add_constant(factor_train_returns)
model = sm.OLS(portfolio_train_series, X).fit()

print("--- Factor Model Regression Results ---")
print(model.summary())
print("\n")

# Extract Betas (Sensitivities)
# We drop the constant (alpha) for the simulation to focus on systematic risk projection
betas = model.params.drop('const')

# 3. Monte Carlo Simulation of Factors
def simulate_factor_returns(factor_data, num_sims):
    """
    Simulate future factor returns using a Multivariate Normal Distribution
    based on historical mean and covariance matrix.
    """
    mu = factor_data.mean()
    cov = factor_data.cov()
    
    # Generate random scenarios
    # Result matrix: [Simulations x Number of Factors]
    simulated_factors = np.random.multivariate_normal(mu, cov, num_sims)
    return pd.DataFrame(simulated_factors, columns=factor_data.columns)

# 1. Simulate Factor Movements
simulated_factors = simulate_factor_returns(factor_train_returns, SIMULATIONS)

# 2. Map Factors to Portfolio Returns using calculated Betas
# Simulated Portfolio Return = Sum(Beta_i * Simulated_Factor_i)
simulated_portfolio_returns = simulated_factors.dot(betas)

# 3. Calculate VaR (Value at Risk)
# For 95% Confidence, we locate the 5th percentile of the simulated distribution
var_95 = np.percentile(simulated_portfolio_returns, (1 - CONFIDENCE_LEVEL) * 100)

print(f"--- Monte Carlo Simulation Results ---")
print(f"Confidence Level: {CONFIDENCE_LEVEL:.0%}")
print(f"1-Day VaR Estimate: {var_95:.4%}")

# Visualization: VaR Distribution
plt.figure(figsize=(10, 6))
plt.hist(simulated_portfolio_returns, bins=50, alpha=0.7, color='#1f77b4', density=True, label='Simulated Returns')
plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.2%}')
plt.title(f'Monte Carlo Simulated Portfolio Return Distribution (95% Confidence)')
plt.xlabel('Daily Log Return')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. Backtesting (Basel Traffic Light)
# Calculate Actual Portfolio Returns in the Test Period
port_test_returns = test_data[TICKERS]
actual_portfolio_test_returns = port_test_returns.dot(WEIGHTS)

def run_backtest(actual_returns, va_r_limit, confidence):
    """
    Compare actual returns against the estimated VaR.
    Categorize model based on exception counts.
    """
    # 1. Identify violations (Actual Return < VaR Limit)
    # Note: Returns and VaR are negative numbers in loss scenarios.
    violations = actual_returns < va_r_limit
    num_violations = violations.sum()
    total_days = len(actual_returns)
    violation_rate = num_violations / total_days
    
    # 2. Determine Zone
    # Standard Basel rules are for 99% (Expected exceptions ~2.5 per 250 days).
    # For 95%, expected exceptions are ~12.5 per 250 days (5%).
    # We adjust the thresholds simply for this 95% context:
    expected_exceptions = total_days * (1 - confidence)
    
    if num_violations <= expected_exceptions * 1.25:
        zone = "Green (Model Accepted)"
    elif num_violations <= expected_exceptions * 1.6:
        zone = "Yellow (Monitor / Warning)"
    else:
        zone = "Red (Model Rejected)"

    return {
        "Total Days": total_days,
        "Violations": num_violations,
        "Expected Violations": int(expected_exceptions),
        "Violation Rate": violation_rate,
        "Zone": zone,
        "Violation Series": violations
    }

# Execute Backtest
backtest_res = run_backtest(actual_portfolio_test_returns, var_95, CONFIDENCE_LEVEL)

print("\n--- Backtesting Results (Out-of-Sample) ---")
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"Total Observations: {backtest_res['Total Days']}")
print(f"Actual Violations: {backtest_res['Violations']}")
print(f"Expected Violations (~5%): {backtest_res['Expected Violations']}")
print(f"Violation Rate: {backtest_res['Violation Rate']:.2%}")
print(f"Model Status: {backtest_res['Zone']}")

# Visualization: Backtest Timeline
plt.figure(figsize=(14, 6))
plt.plot(actual_portfolio_test_returns.index, actual_portfolio_test_returns, label='Actual Daily Returns', alpha=0.6, color='blue', linewidth=1)
plt.axhline(var_95, color='red', linestyle='-', linewidth=2, label=f'Static 95% VaR ({var_95:.2%})')

# Highlight violations
violation_dates = actual_portfolio_test_returns[backtest_res['Violation Series']].index
violation_values = actual_portfolio_test_returns[backtest_res['Violation Series']]
plt.scatter(violation_dates, violation_values, color='red', zorder=5, label='Violations', marker='x', s=60)

plt.title('VaR Backtesting: Actual Returns vs Predicted Beta-Based VaR (95%)')
plt.ylabel('Daily Log Return')
plt.xlabel('Date')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.show()