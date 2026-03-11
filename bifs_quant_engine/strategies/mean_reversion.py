"""Mean Reversion Strategy.

Implements a classic mean reversion strategy using RSI and Bollinger Bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from bifs_quant_engine.strategies.base import Strategy

@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy.
    
    Attributes:
        rsi_period: Period for RSI calculation
        rsi_overbought: RSI threshold for overbought (sell)
        rsi_oversold: RSI threshold for oversold (buy)
        bb_period: Period for Bollinger Bands
        bb_std: Number of standard deviations for Bollinger Bands
        max_position_pct: Maximum position size as % of capital
    """
    rsi_period: int = 14
    rsi_overbought: float = 65.0
    rsi_oversold: float = 35.0
    bb_period: int = 20
    bb_std: float = 2.0
    max_position_pct: float = 0.20


class MeanReversionStrategy(Strategy):
    """Mean Reversion strategy.
    
    Trades based on:
    1. RSI divergence (Overbought/Oversold)
    2. Bollinger Band breakouts
    """
    
    def __init__(
        self,
        name: str,
        universe: List[str],
        config: Optional[MeanReversionConfig] = None,
    ) -> None:
        super().__init__(name)
        self.universe = universe
        self.config = config or MeanReversionConfig()
        
    def generate_target_weights(
        self,
        date: datetime,
        price_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """Generate target weights based on Mean Reversion signals."""
        weights: Dict[str, float] = {}
        
        # Need enough history
        min_history = max(self.config.rsi_period, self.config.bb_period) + 1
        if len(price_history) < min_history:
            return {s: 0.0 for s in self.universe}
            
        for symbol in self.universe:
            # Check if symbol is in price history columns
            if symbol not in price_history.columns:
                continue
                
            prices = price_history[symbol].dropna()
            
            # Ensure we have enough data for this specific symbol
            if len(prices) < min_history:
                weights[symbol] = 0.0
                continue
                
            # Calculate Indicators
            rsi = self._calculate_rsi(prices)
            bb_upper, bb_lower = self._calculate_bb(prices)
            
            # Get latest values (aligned with 'date' by definition of price_history)
            current_price = prices.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_upper = bb_upper.iloc[-1]
            current_lower = bb_lower.iloc[-1]
            
            # Score-based signal: either indicator can trigger,
            # both together give a stronger position
            long_score = 0.0
            short_score = 0.0

            # RSI component
            if current_rsi < self.config.rsi_oversold:
                long_score += 1.0
            elif current_rsi > self.config.rsi_overbought:
                short_score += 1.0

            # Bollinger Band component
            if current_price < current_lower:
                long_score += 1.0
            elif current_price > current_upper:
                short_score += 1.0

            # Convert score to signal (score 1 = half position, score 2 = full)
            if long_score > 0:
                signal = long_score / 2.0
            elif short_score > 0:
                signal = -(short_score / 2.0)
            else:
                signal = 0.0

            if signal != 0.0:
                weights[symbol] = signal * self.config.max_position_pct
            else:
                weights[symbol] = 0.0
                
        return weights

    def _calculate_rsi(self, series: pd.Series) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        
        # Make copy to avoid SettingWithCopyWarning if any
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Use Simple Moving Average for RSI (classic Wilder uses Exponential)
        # Using mean() for simplicity as per original code, can upgrade later
        avg_gain = gain.rolling(window=self.config.rsi_period).mean()
        avg_loss = loss.rolling(window=self.config.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_bb(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = series.rolling(window=self.config.bb_period).mean()
        std = series.rolling(window=self.config.bb_period).std()
        
        upper = sma + (std * self.config.bb_std)
        lower = sma - (std * self.config.bb_std)
        return upper, lower
