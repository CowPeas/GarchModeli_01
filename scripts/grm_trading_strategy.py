"""GRM Trading Strategy - Complete Trading Implementation.

This module provides a complete trading strategy using GRM models,
including backtesting, risk management, and performance analysis.

Example Usage:
    from scripts.grm_trading_strategy import GRMTradingStrategy
    
    strategy = GRMTradingStrategy(
        ticker='BTC-USD',
        initial_capital=100000,
        max_position_pct=0.1
    )
    strategy.prepare_data(start_date='2020-01-01')
    strategy.train_model(train_pct=0.7)
    results = strategy.backtest()
    print(strategy.get_performance_metrics())
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _import_module_directly(module_name: str, file_path: str):
    """Import a module directly without loading __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import specific modules directly to avoid optional dependencies (torch)
_models_dir = PROJECT_ROOT / "models"

_real_data_loader = _import_module_directly(
    "real_data_loader_ts", str(_models_dir / "real_data_loader.py")
)
_baseline_model = _import_module_directly(
    "baseline_model_ts", str(_models_dir / "baseline_model.py")
)
_adaptive_grm = _import_module_directly(
    "adaptive_grm_ts", str(_models_dir / "adaptive_grm.py")
)
_grm_feature_engineering = _import_module_directly(
    "grm_feature_engineering_ts", str(_models_dir / "grm_feature_engineering.py")
)
_gmm_regime_detector = _import_module_directly(
    "gmm_regime_detector_ts", str(_models_dir / "gmm_regime_detector.py")
)

RealDataLoader = _real_data_loader.RealDataLoader
BaselineARIMA = _baseline_model.BaselineARIMA
AdaptiveAlphaGRM = _adaptive_grm.AdaptiveAlphaGRM
GRMFeatureEngineer = _grm_feature_engineering.GRMFeatureEngineer
GMMRegimeDetector = _gmm_regime_detector.GMMRegimeDetector


class GRMTradingStrategy:
    """Complete trading strategy using GRM models.

    Features:
    - Volatility-adaptive position sizing
    - Regime-aware signal generation
    - Risk management integration
    - Performance tracking

    Parameters
    ----------
    ticker : str
        Asset ticker symbol.
    initial_capital : float
        Starting capital in USD.
    max_position_pct : float
        Maximum position size as fraction of capital.
    stop_loss_pct : float
        Stop loss percentage.

    Examples
    --------
    >>> strategy = GRMTradingStrategy(ticker='BTC-USD')
    >>> strategy.prepare_data(start_date='2020-01-01')
    >>> strategy.train_model()
    >>> results = strategy.backtest()
    >>> print(strategy.get_performance_metrics())
    """

    def __init__(
        self,
        ticker: str,
        initial_capital: float = 100000,
        max_position_pct: float = 0.1,
        stop_loss_pct: float = 0.02
    ):
        """Initialize trading strategy."""
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct

        # Model components
        self.data_loader = RealDataLoader(data_source='yahoo')
        self.baseline = None
        self.grm_model = None

        # Data
        self.df = None
        self.returns = None
        self.prices = None
        self.residuals = None
        self.train_size = 0

        # Trading state
        self.position = 0  # Current position size (shares)
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
        self.results = None

    def prepare_data(
        self,
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None
    ) -> 'GRMTradingStrategy':
        """Load and prepare data for trading.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str, optional
            End date. Defaults to today.

        Returns
        -------
        self
            For method chaining.
        """
        self.df, _ = self.data_loader.load_yahoo_finance(
            ticker=self.ticker,
            start_date=start_date,
            end_date=end_date,
            verify_ssl=False
        )

        self.returns = self.df['returns'].values
        self.prices = self.df['Close'].values

        print(f"Loaded {len(self.df)} observations for {self.ticker}")
        return self

    def train_model(self, train_pct: float = 0.7) -> 'GRMTradingStrategy':
        """Train GRM model on historical data.

        Parameters
        ----------
        train_pct : float
            Percentage of data for training.

        Returns
        -------
        self
            For method chaining.
        """
        if self.returns is None:
            raise ValueError("No data loaded. Call prepare_data first.")

        train_size = int(len(self.returns) * train_pct)
        train_returns = self.returns[:train_size]

        # Fit baseline
        self.baseline = BaselineARIMA()
        self.baseline.fit(train_returns, order=(1, 0, 1))

        baseline_pred = self.baseline.predict(steps=len(train_returns))
        train_residuals = train_returns - baseline_pred

        # Extract features and detect regimes
        features = GRMFeatureEngineer.extract_regime_features(
            train_residuals, window=20
        )
        gmm = GMMRegimeDetector(n_components=10, random_state=42)
        self.regime_labels = gmm.fit_predict(features)

        # Create Adaptive GRM (simpler, no torch dependency)
        self.grm_model = AdaptiveAlphaGRM(
            base_alpha=2.0,
            beta=0.1,
            window_size=20,
            alpha_range=(0.5, 5.0),
            volatility_window=50,
            adaptation_speed=0.5
        )

        self.grm_model.fit(train_residuals[20:])

        self.train_size = train_size
        print(f"Model trained on {train_size} observations")
        return self

    def generate_signal(self, current_idx: int) -> Dict:
        """Generate trading signal for current time.

        Parameters
        ----------
        current_idx : int
            Current data index.

        Returns
        -------
        dict
            Signal with direction, strength, and metadata.
        """
        # Get residuals up to current time
        current_returns = self.returns[:current_idx + 1]
        baseline_pred = self.baseline.predict(steps=len(current_returns))
        residuals = current_returns - baseline_pred

        # GRM prediction
        _, correction, final_pred, regime = self.grm_model.predict(
            residuals,
            current_time=current_idx,
            baseline_pred=baseline_pred[-1]
        )

        # Calculate signal strength (prediction relative to volatility)
        window_start = max(0, current_idx - 20)
        recent_vol = np.std(self.returns[window_start:current_idx])

        if recent_vol > 0:
            signal_strength = abs(final_pred) / recent_vol
        else:
            signal_strength = 0

        return {
            'direction': 1 if final_pred > 0 else -1,
            'strength': min(signal_strength, 1.0),
            'prediction': final_pred,
            'regime': regime,
            'volatility': recent_vol
        }

    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal and volatility.

        Uses volatility-adjusted position sizing:
        - Higher volatility → Smaller position
        - Stronger signal → Larger position

        Parameters
        ----------
        signal : dict
            Trading signal from generate_signal.

        Returns
        -------
        float
            Position size in USD.
        """
        base_position = self.capital * self.max_position_pct

        # Volatility adjustment (inverse relationship)
        vol_factor = 0.02 / (signal['volatility'] + 0.01)
        vol_factor = min(max(vol_factor, 0.5), 2.0)  # Limit adjustment

        # Signal strength adjustment
        strength_factor = signal['strength']

        position_size = base_position * vol_factor * strength_factor

        return min(position_size, self.capital * self.max_position_pct)

    def backtest(self, start_idx: Optional[int] = None) -> pd.DataFrame:
        """Run backtest on test data.

        Parameters
        ----------
        start_idx : int, optional
            Starting index for backtest. Defaults to after training.

        Returns
        -------
        pd.DataFrame
            Backtest results with daily metrics.
        """
        if self.grm_model is None:
            raise ValueError("Model not trained. Call train_model first.")

        if start_idx is None:
            start_idx = self.train_size + 20  # Skip first 20 for features

        results = []

        for idx in range(start_idx, len(self.returns)):
            signal = self.generate_signal(idx)

            # Trading logic
            current_price = self.prices[idx]

            # Check stop loss
            if self.position != 0:
                pnl_pct = (
                    (current_price - self.entry_price) / self.entry_price
                )
                stop_hit = (
                    (self.position > 0 and pnl_pct < -self.stop_loss_pct) or
                    (self.position < 0 and pnl_pct > self.stop_loss_pct)
                )
                if stop_hit:
                    self._close_position(idx, current_price, 'stop_loss')

            # New position logic
            if self.position == 0:
                # Enter position if signal is strong enough
                if signal['strength'] > 0.3:  # Threshold for entry
                    position_size = self.calculate_position_size(signal)
                    self._open_position(
                        idx,
                        current_price,
                        signal['direction'],
                        position_size
                    )
            else:
                # Check for exit signal (opposite direction)
                opposite = (
                    (self.position > 0 and signal['direction'] < 0) or
                    (self.position < 0 and signal['direction'] > 0)
                )
                if opposite and signal['strength'] > 0.5:
                    self._close_position(idx, current_price, 'signal_reversal')

            # Track equity
            unrealized_pnl = 0
            if self.position != 0:
                unrealized_pnl = self.position * (current_price - self.entry_price)

            results.append({
                'date': self.df.index[idx],
                'price': current_price,
                'signal': signal['direction'],
                'strength': signal['strength'],
                'position': self.position,
                'capital': self.capital,
                'equity': self.capital + unrealized_pnl
            })

        self.results = pd.DataFrame(results)
        self.equity_curve = self.results['equity'].values

        return self.results

    def _open_position(
        self,
        idx: int,
        price: float,
        direction: int,
        size: float
    ):
        """Open a new position."""
        shares = (size / price) * direction
        self.position = shares
        self.entry_price = price

        self.trades.append({
            'type': 'OPEN',
            'date': self.df.index[idx],
            'price': price,
            'shares': shares,
            'direction': 'LONG' if direction > 0 else 'SHORT'
        })

    def _close_position(self, idx: int, price: float, reason: str):
        """Close current position."""
        pnl = self.position * (price - self.entry_price)
        self.capital += pnl

        self.trades.append({
            'type': 'CLOSE',
            'date': self.df.index[idx],
            'price': price,
            'shares': -self.position,
            'pnl': pnl,
            'reason': reason
        })

        self.position = 0
        self.entry_price = 0

    def get_performance_metrics(self) -> Dict:
        """Calculate strategy performance metrics.

        Returns
        -------
        dict
            Performance metrics including return, Sharpe, drawdown.
        """
        if len(self.equity_curve) == 0:
            return {}

        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]

        # Calculate metrics
        total_return = (
            (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]
        )
        annual_return = total_return * (252 / len(returns))
        volatility = np.std(returns) * np.sqrt(252)

        if volatility > 0:
            sharpe = annual_return / volatility
        else:
            sharpe = 0

        # Max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        max_drawdown = np.max(drawdown)

        # Trade statistics
        closed_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]

        total_closed = len(closed_trades)
        win_rate = len(winning_trades) / max(total_closed, 1)

        return {
            'total_return': f"{total_return * 100:.2f}%",
            'annual_return': f"{annual_return * 100:.2f}%",
            'volatility': f"{volatility * 100:.2f}%",
            'sharpe_ratio': f"{sharpe:.2f}",
            'max_drawdown': f"{max_drawdown * 100:.2f}%",
            'total_trades': total_closed,
            'win_rate': f"{win_rate * 100:.1f}%",
            'final_capital': f"${self.capital:,.2f}"
        }

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame.

        Returns
        -------
        pd.DataFrame
            Trade history with all transactions.
        """
        if len(self.trades) == 0:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def reset(self):
        """Reset strategy to initial state."""
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
        self.results = None


class GRMRiskManager:
    """Risk management for GRM-based trading.

    Features:
    - Dynamic position sizing based on volatility
    - Regime-aware risk limits
    - Portfolio risk monitoring

    Parameters
    ----------
    max_portfolio_risk : float
        Maximum daily portfolio risk (VaR).
    max_position_risk : float
        Maximum risk per position.
    max_drawdown : float
        Maximum allowed drawdown.
    """

    def __init__(
        self,
        max_portfolio_risk: float = 0.02,
        max_position_risk: float = 0.01,
        max_drawdown: float = 0.10
    ):
        """Initialize risk manager."""
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_drawdown = max_drawdown

    def calculate_position_size(
        self,
        signal: Dict,
        capital: float,
        current_volatility: float
    ) -> float:
        """Calculate risk-adjusted position size.

        Uses Kelly Criterion modified with GRM confidence.

        Parameters
        ----------
        signal : dict
            Trading signal with strength and direction.
        capital : float
            Current capital.
        current_volatility : float
            Current market volatility.

        Returns
        -------
        float
            Recommended position size.
        """
        # Estimate win probability from signal strength
        p = 0.5 + signal['strength'] * 0.2  # 50-70% based on strength
        q = 1 - p

        # Assume 1:1 risk/reward for simplicity
        b = 1.0

        # Kelly fraction
        kelly = (p * b - q) / b

        # Half-Kelly for safety
        kelly = kelly * 0.5

        # Volatility adjustment
        vol_adjustment = 0.02 / current_volatility
        vol_adjustment = min(max(vol_adjustment, 0.5), 1.5)

        # Final position size
        position = capital * kelly * vol_adjustment

        # Apply max position limit
        max_position = capital * self.max_position_risk / current_volatility
        position = min(position, max_position)

        return max(position, 0)

    def check_risk_limits(
        self,
        current_equity: float,
        peak_equity: float,
        daily_pnl: float,
        capital: float
    ) -> Dict:
        """Check if current risk is within limits.

        Parameters
        ----------
        current_equity : float
            Current portfolio equity.
        peak_equity : float
            Peak portfolio equity.
        daily_pnl : float
            Today's profit/loss.
        capital : float
            Starting capital.

        Returns
        -------
        dict
            Risk status and any required actions.
        """
        drawdown = (peak_equity - current_equity) / peak_equity
        daily_risk = abs(daily_pnl) / capital

        status = {
            'within_limits': True,
            'current_drawdown': drawdown,
            'daily_risk': daily_risk,
            'actions': []
        }

        if drawdown > self.max_drawdown:
            status['within_limits'] = False
            status['actions'].append('REDUCE_ALL_POSITIONS')
            status['actions'].append('HALT_NEW_TRADES')

        if daily_risk > self.max_portfolio_risk:
            status['actions'].append('REDUCE_POSITION_SIZE')

        return status


if __name__ == "__main__":
    print("=" * 60)
    print("  GRM Trading Strategy Demo")
    print("=" * 60)

    # Initialize strategy
    strategy = GRMTradingStrategy(
        ticker='BTC-USD',
        initial_capital=100000,
        max_position_pct=0.1,
        stop_loss_pct=0.03
    )

    # Prepare and train
    print("\n1. Preparing data...")
    strategy.prepare_data(start_date='2022-01-01')

    print("\n2. Training model...")
    strategy.train_model(train_pct=0.7)

    # Run backtest
    print("\n3. Running backtest...")
    results = strategy.backtest()

    # Print performance
    metrics = strategy.get_performance_metrics()
    print("\n" + "=" * 60)
    print("  STRATEGY PERFORMANCE")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Print trade summary
    trades = strategy.get_trade_history()
    print(f"\n  Trade History: {len(trades)} transactions")
    if len(trades) > 0:
        print(trades.tail(10).to_string())

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
