"""Cointegration analysis utilities.

Provides functions for testing cointegration and calculating
spread/z-score for pairs trading strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CointegrationResult:
    """Result of cointegration test.
    
    Attributes:
        is_cointegrated: Whether series are cointegrated at significance level
        adf_statistic: ADF test statistic
        p_value: P-value of the test
        critical_values: Critical values at 1%, 5%, 10%
        hedge_ratio: Estimated hedge ratio (beta)
    """
    is_cointegrated: bool
    adf_statistic: float
    p_value: float
    critical_values: dict
    hedge_ratio: float


def check_cointegration(
    series1: pd.Series,
    series2: pd.Series,
    significance: float = 0.05,
) -> CointegrationResult:
    """Test for cointegration between two price series.
    
    Uses the Engle-Granger two-step method:
    1. Estimate hedge ratio via OLS
    2. Test residuals for stationarity via ADF test
    
    Args:
        series1: First price series (will be dependent variable)
        series2: Second price series (independent variable)
        significance: Significance level for the test
        
    Returns:
        CointegrationResult with test results
    """
    # Align series
    combined = pd.concat([series1, series2], axis=1).dropna()
    y = combined.iloc[:, 0].values
    x = combined.iloc[:, 1].values
    
    # Step 1: OLS regression to get hedge ratio
    hedge_ratio = calculate_hedge_ratio_ols(y, x)
    
    # Calculate spread (residuals)
    spread = y - hedge_ratio * x
    
    # Step 2: ADF test on spread
    adf_stat, p_value, critical_values = adf_test(spread)
    
    # Determine cointegration
    is_cointegrated = p_value < significance
    
    return CointegrationResult(
        is_cointegrated=is_cointegrated,
        adf_statistic=adf_stat,
        p_value=p_value,
        critical_values=critical_values,
        hedge_ratio=hedge_ratio,
    )


def calculate_hedge_ratio(
    series1: pd.Series,
    series2: pd.Series,
    method: str = "ols",
) -> float:
    """Calculate hedge ratio between two series.
    
    Args:
        series1: Dependent series (to hedge)
        series2: Independent series (hedging instrument)
        method: Method to use ('ols' or 'rolling')
        
    Returns:
        Hedge ratio (beta coefficient)
    """
    combined = pd.concat([series1, series2], axis=1).dropna()
    y = combined.iloc[:, 0].values
    x = combined.iloc[:, 1].values
    
    return calculate_hedge_ratio_ols(y, x)


def calculate_hedge_ratio_ols(y: np.ndarray, x: np.ndarray) -> float:
    """Calculate hedge ratio using OLS regression.
    
    Args:
        y: Dependent variable
        x: Independent variable
        
    Returns:
        Beta coefficient
    """
    # Add constant for intercept
    x_with_const = np.column_stack([np.ones(len(x)), x])
    
    # OLS: beta = (X'X)^-1 X'y
    try:
        coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        return float(coeffs[1])  # Return slope, not intercept
    except np.linalg.LinAlgError:
        return 1.0  # Default to 1:1 hedge


def calculate_spread(
    series1: pd.Series,
    series2: pd.Series,
    hedge_ratio: Optional[float] = None,
) -> pd.Series:
    """Calculate spread between two series.
    
    Spread = series1 - hedge_ratio * series2
    
    Args:
        series1: First price series
        series2: Second price series
        hedge_ratio: Hedge ratio (calculated if not provided)
        
    Returns:
        Spread series
    """
    if hedge_ratio is None:
        hedge_ratio = calculate_hedge_ratio(series1, series2)
    
    combined = pd.concat([series1, series2], axis=1).dropna()
    spread = combined.iloc[:, 0] - hedge_ratio * combined.iloc[:, 1]
    return spread


def calculate_zscore(
    spread: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Calculate z-score of spread.
    
    Z-score = (spread - mean) / std
    
    Uses rolling window for mean and std calculation.
    
    Args:
        spread: Spread series
        lookback: Lookback period for mean/std
        
    Returns:
        Z-score series
    """
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def adf_test(
    series: np.ndarray,
    max_lags: Optional[int] = None,
) -> Tuple[float, float, dict]:
    """Perform Augmented Dickey-Fuller test for stationarity.
    
    Simplified implementation without statsmodels dependency.
    
    Args:
        series: Time series to test
        max_lags: Maximum lags to include
        
    Returns:
        Tuple of (adf_statistic, p_value, critical_values)
    """
    n = len(series)
    if max_lags is None:
        max_lags = int(np.floor(np.power(n - 1, 1/3)))
    
    # Calculate differences
    diff = np.diff(series)
    
    # Lagged level
    lag_level = series[:-1]
    
    # Simple ADF regression: Δy_t = α + β*y_{t-1} + ε_t
    # Test statistic is t-statistic for β
    x = np.column_stack([np.ones(len(lag_level)), lag_level])
    
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(x, diff, rcond=None)
        
        beta = coeffs[1]
        
        # Calculate standard error
        if len(residuals) > 0:
            mse = residuals[0] / (len(diff) - 2)
        else:
            mse = np.sum((diff - x @ coeffs) ** 2) / (len(diff) - 2)
        
        xtx_inv = np.linalg.inv(x.T @ x)
        se_beta = np.sqrt(mse * xtx_inv[1, 1])
        
        adf_stat = beta / se_beta if se_beta > 0 else 0.0
        
    except (np.linalg.LinAlgError, ValueError):
        adf_stat = 0.0
    
    # Approximate critical values for ADF test (no trend, with intercept)
    critical_values = {
        "1%": -3.43,
        "5%": -2.86,
        "10%": -2.57,
    }
    
    # Approximate p-value (simplified)
    if adf_stat < -3.43:
        p_value = 0.01
    elif adf_stat < -2.86:
        p_value = 0.05
    elif adf_stat < -2.57:
        p_value = 0.10
    else:
        p_value = 0.50
    
    return adf_stat, p_value, critical_values


def half_life(spread: pd.Series) -> float:
    """Calculate mean-reversion half-life of spread.
    
    Uses AR(1) model to estimate half-life.
    
    Args:
        spread: Spread series
        
    Returns:
        Half-life in periods
    """
    spread_clean = spread.dropna().values
    if len(spread_clean) < 3:
        return float('inf')
    
    # AR(1): spread_t = α + β * spread_{t-1} + ε
    y = spread_clean[1:]
    x = np.column_stack([np.ones(len(y)), spread_clean[:-1]])
    
    try:
        coeffs = np.linalg.lstsq(x, y, rcond=None)[0]
        beta = coeffs[1]
        
        if beta >= 1:
            return float('inf')
        
        return -np.log(2) / np.log(beta)
    except:
        return float('inf')
