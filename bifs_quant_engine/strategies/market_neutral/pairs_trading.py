"""Pairs trading strategy.

Implements a classic pairs trading strategy using:
- Cointegration-based pair selection
- Z-score entry/exit signals
- Dynamic hedge ratio estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bifs_quant_engine.core.models import Fill
from bifs_quant_engine.strategies.strategy_protocol import (
    BaseStrategy,
    Signal,
    SignalType,
)
from bifs_quant_engine.strategies.market_neutral.cointegration import (
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    check_cointegration,
)


@dataclass
class PairsTradingConfig:
    """Configuration for pairs trading strategy.
    
    Attributes:
        entry_zscore: Z-score threshold for entry (e.g., 2.0)
        exit_zscore: Z-score threshold for exit (e.g., 0.5)
        stop_zscore: Z-score for stop-loss (e.g., 4.0)
        lookback: Lookback period for spread calculation
        hedge_ratio_lookback: Lookback for hedge ratio estimation
        max_position_pct: Maximum position size as % of capital
        cointegration_pvalue: P-value threshold for cointegration
        recalibrate_interval: Days between hedge ratio recalibration
    """
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_zscore: float = 4.0
    lookback: int = 20
    hedge_ratio_lookback: int = 60
    max_position_pct: float = 0.10
    cointegration_pvalue: float = 0.10
    recalibrate_interval: int = 20


@dataclass
class PairState:
    """State for a single pair.
    
    Attributes:
        symbol1: First symbol (long leg)
        symbol2: Second symbol (short leg)
        hedge_ratio: Current hedge ratio
        last_calibration: Date of last calibration
        is_cointegrated: Whether pair is cointegrated
        current_position: Current position (1 = long spread, -1 = short spread)
    """
    symbol1: str
    symbol2: str
    hedge_ratio: float = 1.0
    last_calibration: Optional[datetime] = None
    is_cointegrated: bool = False
    current_position: int = 0  # -1, 0, or 1


class PairsTradingStrategy(BaseStrategy):
    """Pairs trading strategy.
    
    Trades mean-reverting spreads between cointegrated pairs.
    
    Entry:
    - Long spread when z-score < -entry_threshold
    - Short spread when z-score > entry_threshold
    
    Exit:
    - Close when z-score crosses exit_threshold
    - Stop-loss when z-score exceeds stop_threshold
    
    Example:
        strategy = PairsTradingStrategy(
            name="pairs_spy_iwm",
            pairs=[("SPY", "IWM")],
            config=PairsTradingConfig(entry_zscore=2.0),
        )
    """
    
    def __init__(
        self,
        name: str,
        pairs: List[Tuple[str, str]],
        config: Optional[PairsTradingConfig] = None,
    ) -> None:
        """Initialize pairs trading strategy.
        
        Args:
            name: Strategy identifier
            pairs: List of symbol pairs to trade
            config: Strategy configuration
        """
        # Build universe from pairs
        universe = list(set(s for pair in pairs for s in pair))
        super().__init__(name, universe)
        
        self.config = config or PairsTradingConfig()
        self._pairs = pairs
        
        # Track state for each pair
        self._pair_states: Dict[Tuple[str, str], PairState] = {
            pair: PairState(symbol1=pair[0], symbol2=pair[1])
            for pair in pairs
        }
    
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Generate target weights based on z-score signals.
        
        Args:
            date: Current date
            price_history: Historical prices
            
        Returns:
            Target weights for each symbol
        """
        weights: Dict[str, float] = {}
        
        for pair in self._pairs:
            symbol1, symbol2 = pair
            state = self._pair_states[pair]
            
            if symbol1 not in price_history.columns or symbol2 not in price_history.columns:
                continue
            
            series1 = price_history[symbol1].dropna()
            series2 = price_history[symbol2].dropna()
            
            if len(series1) < self.config.hedge_ratio_lookback:
                continue
            
            # Recalibrate hedge ratio periodically
            if self._should_recalibrate(state, date):
                self._calibrate_pair(state, series1, series2, date)
            
            if not state.is_cointegrated:
                continue
            
            # Calculate current z-score
            spread = calculate_spread(series1, series2, state.hedge_ratio)
            zscore = calculate_zscore(spread, self.config.lookback)
            
            if len(zscore) == 0 or pd.isna(zscore.iloc[-1]):
                continue
            
            current_z = zscore.iloc[-1]
            
            # Determine position
            new_position = self._calculate_position(state.current_position, current_z)
            state.current_position = new_position
            
            if new_position != 0:
                # Calculate weights
                # Long spread = long symbol1, short symbol2
                # Short spread = short symbol1, long symbol2
                weight1 = new_position * self.config.max_position_pct
                weight2 = -new_position * state.hedge_ratio * self.config.max_position_pct
                
                weights[symbol1] = weights.get(symbol1, 0) + weight1
                weights[symbol2] = weights.get(symbol2, 0) + weight2
        
        return weights
    
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, Decimal],
    ) -> List[Signal]:
        """Generate trading signals.
        
        Args:
            market_data: Historical prices
            current_positions: Current position quantities
            
        Returns:
            List of trading signals
        """
        signals = []
        
        weights = self.generate_target_weights(datetime.now(), market_data)
        
        for symbol, weight in weights.items():
            signal_type = SignalType.LONG if weight > 0 else SignalType.SHORT
            signals.append(Signal(
                strategy_id=self.name,
                symbol=symbol,
                signal_type=signal_type,
                weight=weight,
            ))
        
        return signals
    
    def get_pair_states(self) -> Dict[Tuple[str, str], PairState]:
        """Get current state of all pairs."""
        return dict(self._pair_states)
    
    def _should_recalibrate(self, state: PairState, date: datetime) -> bool:
        """Check if pair should be recalibrated."""
        if state.last_calibration is None:
            return True
        
        days_since = (date - state.last_calibration).days
        return days_since >= self.config.recalibrate_interval
    
    def _calibrate_pair(
        self,
        state: PairState,
        series1: pd.Series,
        series2: pd.Series,
        date: datetime,
    ) -> None:
        """Calibrate hedge ratio and test cointegration.
        
        Args:
            state: Pair state to update
            series1: Price series for symbol1
            series2: Price series for symbol2
            date: Current date
        """
        # Use recent data for calibration
        lookback = self.config.hedge_ratio_lookback
        s1 = series1.iloc[-lookback:]
        s2 = series2.iloc[-lookback:]
        
        # Test cointegration
        result = check_cointegration(s1, s2, self.config.cointegration_pvalue)
        
        state.is_cointegrated = result.is_cointegrated
        state.hedge_ratio = result.hedge_ratio
        state.last_calibration = date
    
    def _calculate_position(self, current: int, zscore: float) -> int:
        """Determine target position based on z-score.
        
        Args:
            current: Current position (-1, 0, 1)
            zscore: Current z-score
            
        Returns:
            Target position
        """
        # Stop loss
        if abs(zscore) > self.config.stop_zscore:
            return 0
        
        # Exit if crossed exit threshold
        if current != 0:
            if current > 0 and zscore > -self.config.exit_zscore:
                return 0
            elif current < 0 and zscore < self.config.exit_zscore:
                return 0
            return current  # Hold position
        
        # Entry conditions
        if zscore < -self.config.entry_zscore:
            return 1  # Long spread
        elif zscore > self.config.entry_zscore:
            return -1  # Short spread
        
        return 0  # Stay flat
