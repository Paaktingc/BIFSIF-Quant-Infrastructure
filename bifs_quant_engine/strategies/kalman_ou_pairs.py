"""
BIFS Quant Engine — Kalman Filter + Ornstein-Uhlenbeck Optimal Pairs Trading Strategy
======================================================================================

Strategy: Dynamic Pairs Trading with Kalman Filter Hedge Ratio & OU-Optimal Entry/Exit
Target:   Annualised Sharpe Ratio ≈ 2.0–2.5
Author:   BIFSIF Quant Research (Dixon Cheng)
Version:  1.0.0

Architecture
------------
This strategy plugs into the BIFS Quant Engine's Strategy protocol.
It consists of four modules:
    1. PairSelector       — Screens universe for cointegrated, mean-reverting pairs
    2. KalmanHedgeTracker — Dynamically estimates hedge ratio via Kalman Filter
    3. OUParameterFitter  — Fits Ornstein-Uhlenbeck process to the spread
    4. SignalGenerator     — Generates entry/exit signals from OU-optimal thresholds

Key References
--------------
- Ernest Chan, "Algorithmic Trading" (2012) — Kalman Filter pairs on EWA/EWC, Sharpe ~2.4
- Tim Leung & Xin Li, "Optimal Mean Reversion Trading" (2015) — OU optimal stopping
- Huang (2016) — OU pairs on REITs achieving Sharpe 2.3–2.9
- Stübinger & Endres (2019) — Lévy-driven OU on S&P 500, Sharpe 3.92 (HF)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class PairsConfig:
    """Configuration for the pairs trading strategy."""

    # --- Universe & Pair Selection ---
    universe: List[str] = field(default_factory=lambda: [
        # Sector ETFs (good cointegration candidates)
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE",
        # Country ETFs (classic pairs)
        "EWA", "EWC", "EWG", "EWJ", "EWZ", "EWH", "EWT",
        # Fixed Income ETFs
        "TLT", "IEI", "IEF", "SHY", "LQD", "HYG",
        # Commodity-linked
        "GLD", "GDX", "SLV", "USO", "XOP",
    ])

    # --- Pair Selection Thresholds ---
    coint_pvalue: float = 0.05           # Cointegration test p-value threshold
    hurst_threshold: float = 0.5         # Hurst exponent < 0.5 = mean-reverting
    min_halflife: int = 1                # Minimum half-life in trading days
    max_halflife: int = 60               # Maximum half-life in trading days
    min_mean_crossings: int = 12         # Minimum annual mean crossings
    formation_period: int = 252          # Formation window (1 year)
    trading_period: int = 126            # Trading window (6 months)

    # --- Kalman Filter ---
    delta: float = 1e-4                  # Transition covariance multiplier
    observation_covariance: float = 1.0  # Observation noise (R)
    
    # --- OU Process ---
    ou_lookback: int = 120               # Lookback for OU parameter estimation

    # --- Trading Rules ---
    entry_z: float = 1.5                 # Z-score entry threshold (σ from mean)
    exit_z: float = 0.0                  # Z-score exit threshold (at mean)
    stop_loss_z: float = 4.0             # Stop-loss threshold
    max_holding_days: int = 60           # Maximum holding period

    # --- Risk Management ---
    max_pairs: int = 5                   # Maximum simultaneous pairs
    capital_per_pair: float = 0.15       # 15% of capital per pair
    max_portfolio_drawdown: float = 0.10 # 10% portfolio drawdown limit
    transaction_cost_bps: float = 5.0    # Transaction costs in basis points

    # --- CUSUM Regime Detection ---
    cusum_k: float = 0.5                 # CUSUM allowance in standardised innovation units
    cusum_h: float = 5.0                 # CUSUM alarm threshold; breach → cointegration suspect

    # --- Rolling OU Re-estimation ---
    ou_refit_freq: int = 5               # Refit OU parameters every N bars

    # --- Backtest ---
    initial_capital: float = 100_000.0
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"


# =============================================================================
# 2. PAIR SELECTION MODULE
# =============================================================================

class PairSelector:
    """
    Screens a universe of ETFs for cointegrated, mean-reverting pairs.
    
    Selection pipeline (4 filters):
        1. Engle-Granger cointegration test (p < 0.05)
        2. Hurst exponent of spread (H < 0.5)
        3. Half-life of mean reversion (1–60 days)
        4. Mean-crossing frequency (≥ 12/year)
    """

    def __init__(self, config: PairsConfig):
        self.config = config

    def find_cointegrated_pairs(
        self, prices: pd.DataFrame
    ) -> List[Tuple[str, str, float, Optional[float]]]:
        """
        Test all pairs for cointegration using the Johansen trace test.

        Returns list of (ticker1, ticker2, pseudo_pvalue, hedge_ratio) tuples.

        Johansen is preferred over Engle-Granger because it is symmetric
        (invariant to which asset is 'y' vs 'x') and jointly estimates the
        cointegrating vector and stationarity test in one step.
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        tickers = prices.columns.tolist()
        n = len(tickers)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    price_matrix = prices[[tickers[i], tickers[j]]].dropna().values
                    if len(price_matrix) < 50:
                        continue
                    result = coint_johansen(price_matrix, det_order=0, k_ar_diff=1)
                    trace_stat = result.lr1[0]
                    cv_90, cv_95, cv_99 = result.cvt[0]

                    # Map trace statistic to pseudo p-value for ranking
                    if trace_stat > cv_99:
                        pvalue = 0.01
                    elif trace_stat > cv_95:
                        pvalue = 0.05
                    elif trace_stat > cv_90:
                        pvalue = 0.10
                    else:
                        continue  # Not cointegrated at 90%

                    if pvalue > self.config.coint_pvalue:
                        continue

                    # Extract hedge ratio from first cointegrating vector
                    # Convention: v @ [p1, p2] = 0  →  spread = p1 - (-v[1]/v[0]) * p2
                    v = result.evec[:, 0]
                    if abs(v[0]) < 1e-10:
                        continue
                    hedge_ratio = float(-v[1] / v[0])

                    pairs.append((tickers[i], tickers[j], pvalue, hedge_ratio))
                except Exception:
                    continue

        pairs.sort(key=lambda x: x[2])
        return pairs

    @staticmethod
    def compute_hurst_exponent(series: pd.Series) -> float:
        """
        Compute the Hurst exponent using the R/S (rescaled range) method.
        H < 0.5 → mean-reverting | H = 0.5 → random walk | H > 0.5 → trending
        """
        ts = series.dropna().values
        n = len(ts)
        if n < 20:
            return 0.5  # insufficient data

        max_k = min(n // 2, 100)
        lags = range(2, max_k)
        tau = []
        rs_values = []

        for lag in lags:
            # Split into non-overlapping sub-series
            n_subseries = n // lag
            if n_subseries < 1:
                break

            rs_list = []
            for k in range(n_subseries):
                subseries = ts[k * lag:(k + 1) * lag]
                mean_sub = np.mean(subseries)
                deviations = np.cumsum(subseries - mean_sub)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(subseries, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                tau.append(lag)
                rs_values.append(np.mean(rs_list))

        if len(tau) < 2:
            return 0.5

        # Fit log-log regression
        log_tau = np.log(tau)
        log_rs = np.log(rs_values)
        hurst = np.polyfit(log_tau, log_rs, 1)[0]
        return float(np.clip(hurst, 0.0, 1.0))

    @staticmethod
    def compute_halflife(spread: pd.Series) -> float:
        """
        Compute the half-life of mean reversion via AR(1) regression.
        
        spread_t = α + β * spread_{t-1} + ε
        half-life = -ln(2) / ln(β)
        """
        spread_clean = spread.dropna()
        lag = spread_clean.shift(1)
        delta = spread_clean - lag
        lag = lag.iloc[1:]
        delta = delta.iloc[1:]

        if len(lag) < 10:
            return float("inf")

        # OLS regression: Δspread = α + β * spread_{t-1}
        X = np.column_stack([np.ones(len(lag)), lag.values])
        y = delta.values
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            if beta[1] >= 0:
                return float("inf")  # Not mean-reverting
            halflife = -np.log(2) / beta[1]
            return float(halflife)
        except Exception:
            return float("inf")

    @staticmethod
    def count_mean_crossings(spread: pd.Series, window: int = 252) -> int:
        """Count how many times the spread crosses its rolling mean per year."""
        mean = spread.rolling(window=min(window, len(spread) // 2)).mean()
        centered = spread - mean
        centered = centered.dropna()
        crossings = ((centered.iloc[:-1].values * centered.iloc[1:].values) < 0).sum()
        years = len(spread) / 252
        return int(crossings / max(years, 0.5))

    def select_pairs(self, prices: pd.DataFrame) -> List[Dict]:
        """
        Full pair selection pipeline.
        
        Returns ranked list of pair dictionaries with metadata.
        """
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        cointegrated = self.find_cointegrated_pairs(prices)
        qualified_pairs = []

        for ticker1, ticker2, pvalue, johansen_hr in cointegrated:
            # Use Johansen cointegrating vector directly; fall back to OLS if needed
            if johansen_hr is not None:
                hedge_ratio = johansen_hr
            else:
                y = prices[ticker1].values
                x = add_constant(prices[ticker2].values)
                try:
                    result = OLS(y, x).fit()
                    hedge_ratio = result.params[1]
                except Exception:
                    continue

            # Compute spread
            spread = prices[ticker1] - hedge_ratio * prices[ticker2]

            # Hurst exponent (used for scoring, not hard filtering)
            hurst = self.compute_hurst_exponent(spread)

            # Filter 1: Half-life (primary mean-reversion check)
            halflife = self.compute_halflife(spread)
            if not (self.config.min_halflife <= halflife <= self.config.max_halflife):
                continue

            # Filter 2: Mean crossing frequency
            crossings = self.count_mean_crossings(spread)
            if crossings < self.config.min_mean_crossings:
                continue

            qualified_pairs.append({
                "ticker1": ticker1,
                "ticker2": ticker2,
                "coint_pvalue": pvalue,
                "hurst": hurst,
                "halflife": halflife,
                "mean_crossings": crossings,
                "hedge_ratio": hedge_ratio,
            })

        # Rank by composite score: low Hurst + low half-life + high crossings + low p-value
        for pair in qualified_pairs:
            pair["score"] = (
                -pair["hurst"] * 2
                + (1.0 / max(pair["halflife"], 1)) * 10
                + pair["mean_crossings"] * 0.1
                - pair["coint_pvalue"] * 5
            )

        qualified_pairs.sort(key=lambda x: x["score"], reverse=True)
        return qualified_pairs[:self.config.max_pairs]


# =============================================================================
# 3. KALMAN FILTER HEDGE RATIO TRACKER
# =============================================================================

class KalmanHedgeTracker:
    """
    Dynamically estimates the hedge ratio between two assets using a Kalman Filter.
    
    State space model:
        Observation: y_t = [x_t, 1] @ [β_t, α_t]' + ε_t    (linear regression)
        Transition:  θ_t = θ_{t-1} + w_t                      (random walk)
    
    This avoids choosing a lookback window and lets the hedge ratio adapt smoothly.
    """

    def __init__(self, delta: float = 1e-4, obs_cov: float = 1.0,
                 cusum_k: float = 0.5, cusum_h: float = 5.0):
        self.delta = delta
        self.obs_cov = obs_cov
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.theta = None  # State: [slope, intercept]
        self.P = None      # State covariance
        self.initialized = False
        # EM-calibrated noise matrices (set by fit_em; override delta / obs_cov)
        self.Q_em: Optional[np.ndarray] = None
        self.R_em: Optional[float] = None
        # CUSUM state
        self.cusum_upper: float = 0.0
        self.cusum_lower: float = 0.0
        self.cointegration_ok: bool = True

    def initialize(self):
        """Reset Kalman Filter state."""
        self.theta = np.zeros(2)  # [hedge_ratio, intercept]
        self.P = np.eye(2) * 1.0  # Initial uncertainty
        self.cusum_upper = 0.0
        self.cusum_lower = 0.0
        self.cointegration_ok = True
        self.initialized = True

    def update(self, y: float, x: float) -> Tuple[float, float, float]:
        """
        Process one observation and return updated hedge ratio.
        
        Args:
            y: Price of asset 1 (dependent)
            x: Price of asset 2 (independent)
            
        Returns:
            (hedge_ratio, intercept, spread)
        """
        if not self.initialized:
            self.initialize()

        # Observation vector
        F = np.array([x, 1.0])

        # Process and observation noise — use EM-calibrated values when available
        Q = self.Q_em if self.Q_em is not None else self.delta * np.eye(2)
        R = self.R_em if self.R_em is not None else self.obs_cov

        # --- Prediction step ---
        theta_pred = self.theta  # Random walk: θ_t|t-1 = θ_t-1
        P_pred = self.P + Q

        # --- Update step ---
        # Innovation (prediction error)
        y_hat = F @ theta_pred
        e = y - y_hat

        # Innovation covariance
        S = float(F @ P_pred @ F.T + R)

        # Kalman gain
        K = P_pred @ F.T / S

        # State update
        self.theta = theta_pred + K * e

        # Joseph stabilised covariance (numerically positive-definite vs simplified form)
        I_KF = np.eye(2) - np.outer(K, F)
        self.P = I_KF @ P_pred @ I_KF.T + np.outer(K, K) * R

        # --- CUSUM on standardised innovations ---
        e_std = e / np.sqrt(S)
        self.cusum_upper = max(0.0, self.cusum_upper + e_std - self.cusum_k)
        self.cusum_lower = max(0.0, self.cusum_lower - e_std - self.cusum_k)
        self.cointegration_ok = (
            self.cusum_upper < self.cusum_h and self.cusum_lower < self.cusum_h
        )

        hedge_ratio = self.theta[0]
        intercept = self.theta[1]
        spread = e  # The innovation IS the spread

        return hedge_ratio, intercept, spread

    def fit_em(self, y: pd.Series, x: pd.Series, n_iter: int = 10) -> bool:
        """
        Use EM to estimate Q (process noise) and R (observation noise) from data.
        Sets self.Q_em and self.R_em which override delta/obs_cov in update().
        Returns True if EM succeeded, False if pykalman is unavailable.
        """
        try:
            from pykalman import KalmanFilter as PyKF
        except ImportError:
            return False
        try:
            T = len(y)
            obs = y.values.reshape(T, 1)
            obs_matrices = np.column_stack([x.values, np.ones(T)]).reshape(T, 1, 2)
            kf = PyKF(
                n_dim_obs=1,
                n_dim_state=2,
                transition_matrices=np.eye(2),
                observation_matrices=obs_matrices,
                em_vars=["transition_covariance", "observation_covariance"],
            )
            kf = kf.em(obs, n_iter=n_iter)
            self.Q_em = np.array(kf.transition_covariance)
            self.R_em = float(np.array(kf.observation_covariance).flat[0])
            return True
        except Exception:
            return False

    def process_series(self, y: pd.Series, x: pd.Series, fit_em: bool = True) -> pd.DataFrame:
        """
        Process full time series and return DataFrame of Kalman-estimated values.
        If fit_em=True, first calibrates Q and R via EM on the full series.
        """
        if fit_em:
            self.fit_em(y, x)
        self.initialize()

        records = []
        for i in range(len(y)):
            hr, intercept, spread = self.update(y.iloc[i], x.iloc[i])
            records.append({
                "date": y.index[i],
                "hedge_ratio": hr,
                "intercept": intercept,
                "spread": spread,
                "cointegration_ok": self.cointegration_ok,
                "y": y.iloc[i],
                "x": x.iloc[i],
            })

        df = pd.DataFrame(records).set_index("date")
        return df


# =============================================================================
# 4. ORNSTEIN-UHLENBECK PARAMETER FITTER
# =============================================================================

class OUFitter:
    """
    Fits Ornstein-Uhlenbeck process parameters to the spread via MLE.
    
    OU SDE: dX_t = μ(θ − X_t)dt + σ dB_t
    
    Parameters:
        θ (theta): Long-term mean
        μ (mu):    Speed of mean reversion  
        σ (sigma): Instantaneous volatility
    """

    @staticmethod
    def fit(spread: np.ndarray, dt: float = 1.0 / 252) -> Dict[str, float]:
        """
        Maximum Likelihood Estimation of OU parameters.
        
        For discrete observations at interval dt:
            X_{t+1} = θ(1 - e^{-μΔt}) + e^{-μΔt} X_t + ε
            where ε ~ N(0, σ²(1 - e^{-2μΔt}) / (2μ))
        """
        n = len(spread)
        if n < 20:
            return {"theta": 0.0, "mu": 0.0, "sigma": 0.0, "halflife": float("inf")}

        x = spread[:-1]
        y = spread[1:]

        # OLS: y = a + b*x + ε
        X_mat = np.column_stack([np.ones(len(x)), x])
        params = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        a, b = params[0], params[1]
        residuals = y - X_mat @ params
        sigma_e = np.std(residuals, ddof=2)

        # Convert discrete AR(1) to continuous OU
        if b <= 0 or b >= 1:
            # Not valid for OU (non-stationary)
            if b >= 1:
                return {"theta": 0.0, "mu": 0.0, "sigma": 0.0, "halflife": float("inf")}
            # b < 0 means extremely fast reversion
            b = max(b, 0.001)

        mu = -np.log(b) / dt
        theta = a / (1 - b)
        sigma = sigma_e * np.sqrt(-2 * np.log(b) / (dt * (1 - b ** 2)))
        halflife = np.log(2) / mu

        return {
            "theta": float(theta),
            "mu": float(mu),
            "sigma": float(sigma),
            "halflife": float(halflife),
            "ar1_coef": float(b),
        }

    @staticmethod
    def compute_optimal_thresholds(
        mu: float, sigma: float, transaction_cost: float = 0.001
    ) -> Tuple[float, float]:
        """
        Compute OU-optimal entry and exit thresholds.
        
        Based on Bertram (2010): maximise expected return per unit time.
        For simplicity, we use the z-score formulation:
            entry at ±z_entry σ_eq from θ
            exit  at ±z_exit  σ_eq from θ
        
        where σ_eq = σ / √(2μ) is the equilibrium std of the OU process.
        
        A more rigorous approach uses the first-passage time formulation
        from Leung & Li (2015), but this provides a good approximation.
        """
        if mu <= 0 or sigma <= 0:
            return 2.0, 0.0  # fallback defaults

        sigma_eq = sigma / np.sqrt(2 * mu)

        # Optimal entry scales with transaction costs
        # Higher costs → need wider entry to compensate
        cost_adjusted = transaction_cost / sigma_eq if sigma_eq > 0 else 0
        z_entry = max(1.0, 1.5 + cost_adjusted * 5)
        z_exit = 0.0  # Exit at mean (theoretically optimal for symmetric OU)

        return z_entry, z_exit


# =============================================================================
# 5. SIGNAL GENERATOR
# =============================================================================

class PairPosition(Enum):
    FLAT = 0
    LONG_SPREAD = 1   # Long asset1, Short asset2
    SHORT_SPREAD = -1  # Short asset1, Long asset2


@dataclass
class TradeSignal:
    date: pd.Timestamp
    ticker1: str
    ticker2: str
    position: PairPosition
    hedge_ratio: float
    z_score: float
    spread: float
    reason: str


class SignalGenerator:
    """
    Generates trading signals by combining:
        1. Kalman Filter dynamic hedge ratio → stationary spread
        2. OU-fitted z-score → entry/exit thresholds
        3. Risk controls → stop-loss, max holding period
    """

    def __init__(self, config: PairsConfig):
        self.config = config

    def generate_signals(
        self,
        kalman_df: pd.DataFrame,
        ou_params: Dict[str, float],
        ticker1: str,
        ticker2: str,
    ) -> List[TradeSignal]:
        """
        Generate trade signals from Kalman-filtered spread.
        
        The z-score is computed as:
            z = (spread - θ) / σ_eq
        where θ and σ_eq come from the OU fit.
        """
        signals = []
        current_position = PairPosition.FLAT
        days_held = 0

        if ou_params["mu"] <= 0 or ou_params["sigma"] <= 0:
            return signals

        spreads = kalman_df["spread"].values
        tc = self.config.transaction_cost_bps / 10000
        lookback = self.config.ou_lookback
        refit_freq = self.config.ou_refit_freq

        # Mutable OU state — updated every refit_freq bars
        current_ou = ou_params.copy()
        sigma_eq = current_ou["sigma"] / np.sqrt(2 * current_ou["mu"])
        theta = current_ou["theta"]
        z_entry, z_exit = OUFitter.compute_optimal_thresholds(
            current_ou["mu"], current_ou["sigma"], tc
        )
        bars_since_refit = 0

        for i in range(1, len(kalman_df)):
            date = kalman_df.index[i]
            spread = spreads[i]
            hr = kalman_df["hedge_ratio"].iloc[i]
            coint_ok = bool(kalman_df["cointegration_ok"].iloc[i]) \
                if "cointegration_ok" in kalman_df.columns else True

            # --- Rolling OU re-estimation ---
            bars_since_refit += 1
            if bars_since_refit >= refit_freq and i >= lookback:
                window = spreads[max(0, i - lookback):i]
                new_params = OUFitter.fit(window)
                if new_params["mu"] > 0 and new_params["sigma"] > 0:
                    current_ou = new_params
                    sigma_eq = current_ou["sigma"] / np.sqrt(2 * current_ou["mu"])
                    theta = current_ou["theta"]
                    z_entry, z_exit = OUFitter.compute_optimal_thresholds(
                        current_ou["mu"], current_ou["sigma"], tc
                    )
                bars_since_refit = 0

            z = (spread - theta) / sigma_eq if sigma_eq > 0 else 0.0

            # --- CUSUM: exit and suspend if cointegration broken ---
            if not coint_ok:
                if current_position != PairPosition.FLAT:
                    signals.append(TradeSignal(
                        date=date, ticker1=ticker1, ticker2=ticker2,
                        position=PairPosition.FLAT, hedge_ratio=hr,
                        z_score=z, spread=spread,
                        reason="EXIT (CUSUM: cointegration breakdown)",
                    ))
                    current_position = PairPosition.FLAT
                    days_held = 0
                continue

            if current_position == PairPosition.FLAT:
                # --- Entry conditions ---
                if z < -z_entry:
                    # Spread too low → Long spread (buy asset1, sell asset2)
                    current_position = PairPosition.LONG_SPREAD

                    days_held = 0
                    signals.append(TradeSignal(
                        date=date, ticker1=ticker1, ticker2=ticker2,
                        position=PairPosition.LONG_SPREAD, hedge_ratio=hr,
                        z_score=z, spread=spread, reason=f"ENTRY: z={z:.2f} < -{z_entry:.2f}"
                    ))

                elif z > z_entry:
                    # Spread too high → Short spread (sell asset1, buy asset2)
                    current_position = PairPosition.SHORT_SPREAD

                    days_held = 0
                    signals.append(TradeSignal(
                        date=date, ticker1=ticker1, ticker2=ticker2,
                        position=PairPosition.SHORT_SPREAD, hedge_ratio=hr,
                        z_score=z, spread=spread, reason=f"ENTRY: z={z:.2f} > {z_entry:.2f}"
                    ))

            else:
                days_held += 1

                # --- Exit conditions ---
                exit_reason = None

                # 1. Mean reversion exit
                if current_position == PairPosition.LONG_SPREAD and z >= z_exit:
                    exit_reason = f"EXIT (mean reversion): z={z:.2f} >= {z_exit:.2f}"
                elif current_position == PairPosition.SHORT_SPREAD and z <= z_exit:
                    exit_reason = f"EXIT (mean reversion): z={z:.2f} <= {z_exit:.2f}"

                # 2. Stop-loss
                elif abs(z) > self.config.stop_loss_z:
                    exit_reason = f"EXIT (stop-loss): |z|={abs(z):.2f} > {self.config.stop_loss_z}"

                # 3. Max holding period
                elif days_held >= self.config.max_holding_days:
                    exit_reason = f"EXIT (max hold): {days_held} days"

                if exit_reason:
                    signals.append(TradeSignal(
                        date=date, ticker1=ticker1, ticker2=ticker2,
                        position=PairPosition.FLAT, hedge_ratio=hr,
                        z_score=z, spread=spread, reason=exit_reason
                    ))
                    current_position = PairPosition.FLAT
                    days_held = 0

        return signals


# =============================================================================
# 6. BACKTESTER
# =============================================================================

@dataclass
class PairsBacktestResult:
    """Results from a pairs trading backtest."""
    pair: str
    total_return: float
    annualised_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_holding_days: float
    equity_curve: pd.Series
    daily_returns: pd.Series


class PairsBacktester:
    """
    Backtests the Kalman + OU pairs trading strategy with realistic costs.
    """

    def __init__(self, config: PairsConfig):
        self.config = config

    def backtest_pair(
        self,
        prices: pd.DataFrame,
        ticker1: str,
        ticker2: str,
    ) -> Optional[PairsBacktestResult]:
        """Run backtest for a single pair."""
        
        if ticker1 not in prices.columns or ticker2 not in prices.columns:
            return None

        y = prices[ticker1].dropna()
        x = prices[ticker2].dropna()

        # Align dates
        common_idx = y.index.intersection(x.index)
        y = y.loc[common_idx]
        x = x.loc[common_idx]

        if len(y) < self.config.formation_period + 50:
            return None

        # --- Phase 1: Kalman Filter ---
        kalman = KalmanHedgeTracker(
            delta=self.config.delta,
            obs_cov=self.config.observation_covariance,
        )
        kalman_df = kalman.process_series(y, x)

        # Warm up: skip first N observations for Kalman to converge
        warmup = min(60, len(kalman_df) // 4)
        kalman_df = kalman_df.iloc[warmup:]

        if len(kalman_df) < 100:
            return None

        # --- Phase 2: OU Fit (rolling) ---
        spreads = kalman_df["spread"].values
        lookback = self.config.ou_lookback

        if len(spreads) < lookback + 50:
            return None

        # Fit OU on formation period
        formation_spread = spreads[:lookback]
        ou_params = OUFitter.fit(formation_spread)

        if ou_params["mu"] <= 0:
            return None

        # --- Phase 3: Generate Signals on trading period ---
        trading_df = kalman_df.iloc[lookback:]
        signal_gen = SignalGenerator(self.config)
        signals = signal_gen.generate_signals(
            trading_df, ou_params, ticker1, ticker2
        )

        if len(signals) < 2:
            return None

        # --- Phase 4: Compute P&L ---
        daily_returns = pd.Series(0.0, index=trading_df.index)
        tc_bps = self.config.transaction_cost_bps / 10000
        
        position = PairPosition.FLAT
        entry_spread = 0.0
        trades = []
        current_trade_return = 0.0
        holding_days_list = []
        current_hold = 0

        signal_dict = {s.date: s for s in signals}

        for i in range(len(trading_df)):
            date = trading_df.index[i]
            spread = trading_df["spread"].iloc[i]

            if date in signal_dict:
                sig = signal_dict[date]

                if position == PairPosition.FLAT and sig.position != PairPosition.FLAT:
                    # Opening position
                    position = sig.position
                    entry_spread = spread
                    current_hold = 0
                    # Transaction cost on entry
                    daily_returns.iloc[i] -= tc_bps

                elif position != PairPosition.FLAT and sig.position == PairPosition.FLAT:
                    # Closing position
                    pnl = (spread - entry_spread) * position.value
                    # Normalize by spread volatility for stable returns
                    spread_std = np.std(trading_df["spread"].iloc[max(0, i-60):i+1])
                    if spread_std > 0:
                        daily_returns.iloc[i] += pnl / spread_std * 0.01  # Scale factor
                    # Transaction cost on exit
                    daily_returns.iloc[i] -= tc_bps

                    trades.append(pnl)
                    holding_days_list.append(current_hold)
                    position = PairPosition.FLAT

            elif position != PairPosition.FLAT:
                current_hold += 1
                # Mark-to-market daily P&L
                if i > 0:
                    prev_spread = trading_df["spread"].iloc[i - 1]
                    daily_pnl = (spread - prev_spread) * position.value
                    spread_std = np.std(trading_df["spread"].iloc[max(0, i-60):i+1])
                    if spread_std > 0:
                        daily_returns.iloc[i] += daily_pnl / spread_std * 0.01

        # --- Phase 5: Compute Metrics ---
        equity_curve = (1 + daily_returns).cumprod()
        total_return = equity_curve.iloc[-1] - 1
        n_years = len(daily_returns) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

        daily_std = daily_returns.std()
        sharpe = (daily_returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0

        # Max drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()

        # Win rate
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades) if trades else 0

        avg_hold = np.mean(holding_days_list) if holding_days_list else 0

        return PairsBacktestResult(
            pair=f"{ticker1}/{ticker2}",
            total_return=total_return,
            annualised_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            num_trades=len(trades),
            win_rate=win_rate,
            avg_holding_days=avg_hold,
            equity_curve=equity_curve, 
            daily_returns=daily_returns,
        )

    def run_portfolio_backtest(
        self, prices: pd.DataFrame, pairs: List[Dict]
    ) -> Dict:
        """
        Run backtest across multiple pairs and aggregate portfolio performance.
        """
        results = []
        for pair in pairs:
            result = self.backtest_pair(
                prices, pair["ticker1"], pair["ticker2"]
            )
            if result and result.num_trades >= 3:
                results.append(result)

        if not results:
            return {"error": "No valid pair backtests"}

        # Equal-weight portfolio of pair strategies
        all_returns = pd.DataFrame({
            r.pair: r.daily_returns for r in results
        })
        portfolio_returns = all_returns.mean(axis=1)
        portfolio_equity = (1 + portfolio_returns).cumprod()

        total_return = portfolio_equity.iloc[-1] - 1
        n_years = len(portfolio_returns) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1
        daily_std = portfolio_returns.std()
        sharpe = (portfolio_returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0

        running_max = portfolio_equity.cummax()
        max_dd = ((portfolio_equity - running_max) / running_max).min()

        return {
            "portfolio_sharpe": sharpe,
            "portfolio_ann_return": ann_return,
            "portfolio_max_drawdown": max_dd,
            "portfolio_equity_curve": portfolio_equity,
            "num_pairs": len(results),
            "pair_results": results,
            "daily_returns": portfolio_returns,
        }


# =============================================================================
# 7. MAIN RUNNER — PLUG INTO BIFS QUANT ENGINE
# =============================================================================

def run_strategy(config: Optional[PairsConfig] = None):
    """
    Full pipeline:
        1. Download price data
        2. Select pairs
        3. Run Kalman + OU backtest
        4. Report results
    """
    import yfinance as yf

    if config is None:
        config = PairsConfig()

    print("=" * 70)
    print("BIFS Quant Engine — Kalman + OU Optimal Pairs Trading Strategy")
    print("=" * 70)

    # --- Step 1: Download Data ---
    print(f"\n[1/4] Downloading price data for {len(config.universe)} assets...")
    prices = yf.download(
        config.universe,
        start=config.start_date,
        end=config.end_date,
        progress=False,
    )
    # Handle both old ('Adj Close') and new ('Close') yfinance column names
    if isinstance(prices.columns, pd.MultiIndex):
        if "Adj Close" in prices.columns.get_level_values(0):
            prices = prices["Adj Close"]
        else:
            prices = prices["Close"]
    prices = prices.dropna(axis=1, how="any")
    print(f"  → {len(prices.columns)} assets with clean data, {len(prices)} trading days")

    # --- Step 2: Pair Selection ---
    print("\n[2/4] Screening for cointegrated, mean-reverting pairs...")
    selector = PairSelector(config)

    # Use formation period for pair selection
    formation_prices = prices.iloc[:config.formation_period]
    pairs = selector.select_pairs(formation_prices)

    print(f"  → Found {len(pairs)} qualified pairs:")
    for p in pairs:
        print(f"    {p['ticker1']}/{p['ticker2']}  "
              f"(coint p={p['coint_pvalue']:.4f}, "
              f"H={p['hurst']:.3f}, "
              f"HL={p['halflife']:.1f}d, "
              f"crossings={p['mean_crossings']}/yr)")

    if not pairs:
        print("  ⚠ No pairs passed all filters. Try relaxing thresholds.")
        return None

    # --- Step 3: Backtest ---
    print("\n[3/4] Running Kalman + OU backtest...")
    backtester = PairsBacktester(config)
    portfolio = backtester.run_portfolio_backtest(prices, pairs)

    if "error" in portfolio:
        print(f"  ⚠ {portfolio['error']}")
        return None

    # --- Step 4: Results ---
    print("\n[4/4] Results:")
    print("=" * 70)
    print(f"  Portfolio Sharpe Ratio:    {portfolio['portfolio_sharpe']:.2f}")
    print(f"  Annualised Return:         {portfolio['portfolio_ann_return']*100:.1f}%")
    print(f"  Max Drawdown:              {portfolio['portfolio_max_drawdown']*100:.1f}%")
    print(f"  Number of Pairs:           {portfolio['num_pairs']}")
    print()

    for r in portfolio["pair_results"]:
        print(f"  {r.pair:12s}  Sharpe={r.sharpe_ratio:6.2f}  "
              f"Return={r.annualised_return*100:6.1f}%  "
              f"MaxDD={r.max_drawdown*100:6.1f}%  "
              f"Trades={r.num_trades:3d}  "
              f"WinRate={r.win_rate*100:5.1f}%  "
              f"AvgHold={r.avg_holding_days:4.1f}d")

    print("=" * 70)
    return portfolio


# =============================================================================
# 8. BIFS QUANT ENGINE STRATEGY ADAPTER
# =============================================================================

class KalmanOUPairsStrategy:
    """
    Adapter to plug into the BIFS Quant Engine's Strategy protocol.
    
    Usage with BIFS Quant Engine:
    
        from bifs_quant_engine.trading.session import TradingSession, SessionConfig
        from bifs_quant_engine.brokers.paper_broker import PaperBroker, PaperBrokerConfig
        from bifs_quant_engine.strategies.kalman_ou_pairs import KalmanOUPairsStrategy
        
        broker = PaperBroker(PaperBrokerConfig(initial_cash=100000))
        session = TradingSession(broker, SessionConfig())
        session.add_strategy(KalmanOUPairsStrategy())
        session.start()
    """

    def __init__(self, config: Optional[PairsConfig] = None):
        self.config = config or PairsConfig()
        self.name = "Kalman-OU Pairs Trading"
        self.kalman_trackers: Dict[str, KalmanHedgeTracker] = {}
        self.ou_params: Dict[str, Dict] = {}
        self.positions: Dict[str, PairPosition] = {}
        self.selected_pairs: List[Dict] = []

    def on_start(self, universe_prices: pd.DataFrame):
        """Called at strategy start — select pairs and initialize trackers."""
        selector = PairSelector(self.config)
        self.selected_pairs = selector.select_pairs(universe_prices)

        for pair in self.selected_pairs:
            key = f"{pair['ticker1']}/{pair['ticker2']}"
            self.kalman_trackers[key] = KalmanHedgeTracker(
                delta=self.config.delta,
                obs_cov=self.config.observation_covariance,
            )
            self.positions[key] = PairPosition.FLAT

    def on_bar(self, date, prices: Dict[str, float]) -> List[TradeSignal]:
        """Called on each bar — update Kalman filter and generate signals."""
        signals = []

        for pair in self.selected_pairs:
            t1, t2 = pair["ticker1"], pair["ticker2"]
            key = f"{t1}/{t2}"

            if t1 not in prices or t2 not in prices:
                continue

            # Update Kalman filter
            tracker = self.kalman_trackers[key]
            hr, intercept, spread = tracker.update(prices[t1], prices[t2])

            # Compute z-score if OU params available
            if key in self.ou_params:
                ou = self.ou_params[key]
                if ou["mu"] > 0 and ou["sigma"] > 0:
                    sigma_eq = ou["sigma"] / np.sqrt(2 * ou["mu"])
                    z = (spread - ou["theta"]) / sigma_eq if sigma_eq > 0 else 0

                    # Generate signals based on z-score thresholds
                    current = self.positions[key]
                    if current == PairPosition.FLAT:
                        if z < -self.config.entry_z:
                            signals.append(TradeSignal(
                                date=date, ticker1=t1, ticker2=t2,
                                position=PairPosition.LONG_SPREAD,
                                hedge_ratio=hr, z_score=z, spread=spread,
                                reason=f"ENTRY LONG: z={z:.2f}"
                            ))
                            self.positions[key] = PairPosition.LONG_SPREAD
                        elif z > self.config.entry_z:
                            signals.append(TradeSignal(
                                date=date, ticker1=t1, ticker2=t2,
                                position=PairPosition.SHORT_SPREAD,
                                hedge_ratio=hr, z_score=z, spread=spread,
                                reason=f"ENTRY SHORT: z={z:.2f}"
                            ))
                            self.positions[key] = PairPosition.SHORT_SPREAD

                    elif current != PairPosition.FLAT:
                        should_exit = False
                        if current == PairPosition.LONG_SPREAD and z >= self.config.exit_z:
                            should_exit = True
                        elif current == PairPosition.SHORT_SPREAD and z <= self.config.exit_z:
                            should_exit = True
                        elif abs(z) > self.config.stop_loss_z:
                            should_exit = True

                        if should_exit:
                            signals.append(TradeSignal(
                                date=date, ticker1=t1, ticker2=t2,
                                position=PairPosition.FLAT,
                                hedge_ratio=hr, z_score=z, spread=spread,
                                reason=f"EXIT: z={z:.2f}"
                            ))
                            self.positions[key] = PairPosition.FLAT

        return signals

    def refit_ou(self, spread_history: np.ndarray, pair_key: str):
        """Periodically refit OU parameters (call every N bars)."""
        self.ou_params[pair_key] = OUFitter.fit(spread_history)


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    # Use default config — balanced filters for pair selection
    # Override with tighter thresholds only if pairs are plentiful
    config = PairsConfig(
        # OU-optimal thresholds
        entry_z=1.5,
        exit_z=0.0,
        stop_loss_z=3.5,
        max_holding_days=45,
        # Risk
        max_pairs=5,
        transaction_cost_bps=5.0,
        # Backtest period
        start_date="2018-01-01",
        end_date="2024-12-31",
    )

    result = run_strategy(config)
