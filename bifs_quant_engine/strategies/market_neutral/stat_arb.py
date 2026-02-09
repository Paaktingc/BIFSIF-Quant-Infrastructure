"""Statistical arbitrage strategy.

Implements a multi-asset statistical arbitrage strategy using:
- PCA-based factor decomposition
- Mean-reversion on residuals
- Multi-pair trading
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bifs_quant_engine.core.models import Fill
from bifs_quant_engine.strategies.strategy_protocol import (
    BaseStrategy,
    Signal,
    SignalType,
)
from bifs_quant_engine.strategies.market_neutral.cointegration import (
    calculate_zscore,
    half_life,
)


@dataclass
class StatArbConfig:
    """Configuration for statistical arbitrage strategy.
    
    Attributes:
        lookback: Lookback period for calculations
        num_factors: Number of PCA factors to extract
        entry_zscore: Z-score threshold for entry
        exit_zscore: Z-score threshold for exit
        stop_zscore: Z-score for stop-loss
        residual_lookback: Lookback for residual mean/std
        max_position_pct: Maximum position per stock
        min_half_life: Minimum half-life to trade (days)
        max_half_life: Maximum half-life to trade (days)
        rebalance_threshold: Minimum weight change to rebalance
    """
    lookback: int = 60
    num_factors: int = 5
    entry_zscore: float = 1.5
    exit_zscore: float = 0.5
    stop_zscore: float = 3.0
    residual_lookback: int = 20
    max_position_pct: float = 0.05
    min_half_life: float = 2.0
    max_half_life: float = 30.0
    rebalance_threshold: float = 0.01


class StatisticalArbitrageStrategy(BaseStrategy):
    """Statistical arbitrage strategy using PCA factor decomposition.
    
    Strategy:
    1. Extract common factors from returns using PCA
    2. Compute residual returns (idiosyncratic component)
    3. Trade mean-reversion on residuals
    
    Example:
        strategy = StatisticalArbitrageStrategy(
            name="stat_arb_tech",
            universe=["AAPL", "MSFT", "GOOG", "META", "AMZN"],
        )
    """
    
    def __init__(
        self,
        name: str,
        universe: List[str],
        config: Optional[StatArbConfig] = None,
    ) -> None:
        """Initialize strategy.
        
        Args:
            name: Strategy identifier
            universe: Symbols to trade
            config: Strategy configuration
        """
        super().__init__(name, universe)
        self.config = config or StatArbConfig()
        
        # Factor model state
        self._factor_loadings: Optional[np.ndarray] = None
        self._factor_means: Optional[np.ndarray] = None
        self._last_residuals: Optional[pd.DataFrame] = None
        self._current_positions: Dict[str, int] = {}  # -1, 0, 1
    
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Generate target weights based on residual z-scores.
        
        Args:
            date: Current date
            price_history: Historical prices
            
        Returns:
            Target weights
        """
        weights: Dict[str, float] = {}
        
        # Need enough history
        if len(price_history) < self.config.lookback:
            return weights
        
        # Filter to universe
        available = [s for s in self._universe if s in price_history.columns]
        if len(available) < self.config.num_factors + 1:
            return weights
        
        prices = price_history[available].iloc[-self.config.lookback:]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        if len(returns) < 10:
            return weights
        
        # Fit factor model and get residuals
        residuals = self._calculate_residuals(returns)
        if residuals is None:
            return weights
        
        # Calculate z-scores of residuals
        for symbol in available:
            if symbol not in residuals.columns:
                continue
            
            residual_series = residuals[symbol]
            zscore_series = calculate_zscore(residual_series, self.config.residual_lookback)
            
            if len(zscore_series) == 0 or pd.isna(zscore_series.iloc[-1]):
                continue
            
            current_z = zscore_series.iloc[-1]
            
            # Check half-life
            hl = half_life(residual_series)
            if hl < self.config.min_half_life or hl > self.config.max_half_life:
                continue
            
            # Determine position
            current_pos = self._current_positions.get(symbol, 0)
            new_pos = self._calculate_position(current_pos, current_z)
            self._current_positions[symbol] = new_pos
            
            if new_pos != 0:
                # Weight proportional to inverse half-life (faster mean reversion = higher weight)
                weight_scale = min(1.0, self.config.max_half_life / hl)
                weights[symbol] = -new_pos * self.config.max_position_pct * weight_scale
        
        # Ensure market neutrality
        weights = self._neutralize_weights(weights, prices)
        
        return weights
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Generate trading signals.
        
        Args:
            market_data: Historical prices
            current_positions: Current positions
            
        Returns:
            List of signals
        """
        signals = []
        
        weights = self.generate_target_weights(datetime.now(), market_data)
        
        for symbol, weight in weights.items():
            if abs(weight) < self.config.rebalance_threshold:
                continue
            
            signal_type = SignalType.LONG if weight > 0 else SignalType.SHORT
            signals.append(Signal(
                strategy_id=self.name,
                symbol=symbol,
                signal_type=signal_type,
                weight=weight,
            ))
        
        return signals
    
    def _calculate_residuals(self, returns: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate residual returns after removing common factors.
        
        Uses PCA to extract factors, then calculates residuals.
        
        Args:
            returns: Return series for all assets
            
        Returns:
            Residual returns or None if calculation fails
        """
        # Standardize returns
        mean = returns.mean()
        std = returns.std()
        std = std.replace(0, 1)  # Avoid division by zero
        standardized = (returns - mean) / std
        
        # Fill any remaining NaNs
        standardized = standardized.fillna(0)
        
        # PCA
        try:
            X = standardized.values
            n_components = min(self.config.num_factors, X.shape[1] - 1)
            
            if n_components < 1:
                return None
            
            # Compute covariance matrix
            cov = np.cov(X.T)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Take top n components
            loadings = eigenvectors[:, :n_components]
            
            # Project to factor space and back
            factors = X @ loadings
            reconstructed = factors @ loadings.T
            
            # Residuals = original - reconstructed
            residuals = X - reconstructed
            
            # Convert back to DataFrame
            return pd.DataFrame(
                residuals,
                index=returns.index,
                columns=returns.columns,
            )
            
        except (np.linalg.LinAlgError, ValueError):
            return None
    
    def _calculate_position(self, current: int, zscore: float) -> int:
        """Determine target position.
        
        Args:
            current: Current position
            zscore: Current z-score
            
        Returns:
            Target position
        """
        # Stop loss
        if abs(zscore) > self.config.stop_zscore:
            return 0
        
        # Exit if crossed exit threshold
        if current != 0:
            if current > 0 and zscore < self.config.exit_zscore:
                return 0
            elif current < 0 and zscore > -self.config.exit_zscore:
                return 0
            return current
        
        # Entry conditions (mean reversion: sell high z, buy low z)
        if zscore > self.config.entry_zscore:
            return -1  # Sell overvalued
        elif zscore < -self.config.entry_zscore:
            return 1  # Buy undervalued
        
        return 0
    
    def _neutralize_weights(
        self,
        weights: Dict[str, float],
        prices: pd.DataFrame,
    ) -> Dict[str, float]:
        """Adjust weights to be dollar-neutral.
        
        Args:
            weights: Raw target weights
            prices: Current prices
            
        Returns:
            Neutralized weights
        """
        if not weights:
            return weights
        
        # Calculate net exposure
        long_weight = sum(w for w in weights.values() if w > 0)
        short_weight = sum(w for w in weights.values() if w < 0)
        
        net = long_weight + short_weight
        
        if abs(net) < 0.001:
            return weights
        
        # Adjust to neutralize
        # If net positive, reduce longs or increase shorts
        # If net negative, reduce shorts or increase longs
        
        if net > 0:
            # Reduce longs proportionally
            scale = (long_weight - net) / long_weight if long_weight > 0 else 1
            return {
                s: w * scale if w > 0 else w
                for s, w in weights.items()
            }
        else:
            # Reduce shorts proportionally
            scale = (abs(short_weight) - abs(net)) / abs(short_weight) if short_weight < 0 else 1
            return {
                s: w * scale if w < 0 else w
                for s, w in weights.items()
            }
