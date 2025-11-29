"""GRM Agent Tools - For AI Agent Integration.

This module provides tools for integrating GRM models with AI agents
like LangChain, AutoGen, or custom agent frameworks.

Example Usage:
    from scripts.grm_agent_tools import GRMAgentTools
    
    tools = GRMAgentTools(ticker='BTC-USD')
    tools.tool_load_data(start_date='2020-01-01')
    tools.tool_fit_model()
    signal = tools.tool_get_trading_signal()
    print(signal)
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

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
    "real_data_loader", str(_models_dir / "real_data_loader.py")
)
_baseline_model = _import_module_directly(
    "baseline_model", str(_models_dir / "baseline_model.py")
)
_adaptive_grm = _import_module_directly(
    "adaptive_grm", str(_models_dir / "adaptive_grm.py")
)
_grm_feature_engineering = _import_module_directly(
    "grm_feature_engineering", str(_models_dir / "grm_feature_engineering.py")
)
_gmm_regime_detector = _import_module_directly(
    "gmm_regime_detector", str(_models_dir / "gmm_regime_detector.py")
)

RealDataLoader = _real_data_loader.RealDataLoader
BaselineARIMA = _baseline_model.BaselineARIMA
AdaptiveAlphaGRM = _adaptive_grm.AdaptiveAlphaGRM
GRMFeatureEngineer = _grm_feature_engineering.GRMFeatureEngineer
GMMRegimeDetector = _gmm_regime_detector.GMMRegimeDetector


class GRMAgentTools:
    """Tools for AI agents to interact with GRM models.

    Provides a simple interface for agents to:
    1. Load and analyze market data
    2. Generate volatility forecasts
    3. Detect market regimes
    4. Get trading signals based on GRM predictions

    Parameters
    ----------
    ticker : str
        Asset ticker symbol (e.g., 'BTC-USD', 'ETH-USD', 'SPY').

    Examples
    --------
    >>> tools = GRMAgentTools(ticker='BTC-USD')
    >>> tools.tool_load_data(start_date='2020-01-01')
    >>> tools.tool_fit_model()
    >>> signal = tools.tool_get_trading_signal()
    >>> print(signal['recommendation'])
    """

    def __init__(self, ticker: str = 'BTC-USD'):
        """Initialize GRM tools for a specific asset."""
        self.ticker = ticker
        self.data_loader = RealDataLoader(data_source='yahoo')
        self.baseline = None
        self.grm_model = None
        self.regime_detector = None
        self.is_fitted = False
        self.df = None
        self.returns = None
        self.residuals = None
        self.regime_labels = None

    def tool_load_data(
        self,
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Agent Tool: Load market data for analysis.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str, optional
            End date. Defaults to today.

        Returns
        -------
        dict
            Data summary with key statistics.
        """
        try:
            df, _ = self.data_loader.load_yahoo_finance(
                ticker=self.ticker,
                start_date=start_date,
                end_date=end_date,
                verify_ssl=False
            )

            self.df = df
            self.returns = df['returns'].values

            return {
                'status': 'success',
                'ticker': self.ticker,
                'observations': len(df),
                'date_range': f"{df.index[0]} to {df.index[-1]}",
                'mean_return': float(np.mean(self.returns)),
                'volatility': float(np.std(self.returns)),
                'min_return': float(np.min(self.returns)),
                'max_return': float(np.max(self.returns))
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def tool_fit_model(self) -> Dict[str, Any]:
        """Agent Tool: Fit GRM model on loaded data.

        Returns
        -------
        dict
            Model fitting status and parameters.
        """
        if self.returns is None:
            return {
                'status': 'error',
                'message': 'No data loaded. Call tool_load_data first.'
            }

        try:
            # Fit baseline
            self.baseline = BaselineARIMA()
            self.baseline.fit(self.returns, order=(1, 0, 1))

            baseline_pred = self.baseline.predict(steps=len(self.returns))
            self.residuals = self.returns - baseline_pred

            # Extract features and detect regimes
            features = GRMFeatureEngineer.extract_regime_features(
                self.residuals, window=20
            )
            self.regime_detector = GMMRegimeDetector(
                n_components=10, random_state=42
            )
            self.regime_labels = self.regime_detector.fit_predict(features)

            # Fit GRM
            self.grm_model = AdaptiveAlphaGRM(
                base_alpha=2.0,
                beta=0.1,
                window_size=20,
                alpha_range=(0.5, 5.0)
            )
            # Skip first 20 for features
            self.grm_model.fit(self.residuals[20:])

            self.is_fitted = True

            return {
                'status': 'success',
                'baseline_rmse': float(np.sqrt(np.mean(self.residuals ** 2))),
                'regimes_detected': int(len(np.unique(self.regime_labels))),
                'model_type': 'AdaptiveAlphaGRM'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def tool_get_prediction(self, steps_ahead: int = 1) -> Dict[str, Any]:
        """Agent Tool: Get GRM-enhanced prediction.

        Parameters
        ----------
        steps_ahead : int
            Number of steps to predict ahead.

        Returns
        -------
        dict
            Prediction results with confidence information.
        """
        if not self.is_fitted:
            return {
                'status': 'error',
                'message': 'Model not fitted. Call tool_fit_model first.'
            }

        try:
            current_time = len(self.residuals) - 1
            baseline_pred = self.baseline.predict(steps=steps_ahead)[0]

            _, correction, final_pred, regime = self.grm_model.predict(
                self.residuals,
                current_time=current_time,
                baseline_pred=baseline_pred
            )

            # Get adaptation stats
            stats = self.grm_model.get_adaptation_stats()

            return {
                'status': 'success',
                'baseline_prediction': float(baseline_pred),
                'grm_correction': float(correction),
                'final_prediction': float(final_pred),
                'current_alpha': float(stats.get('mean_alpha', 2.0)),
                'current_volatility': float(stats.get('mean_volatility', 0.0)),
                'current_regime': int(regime),
                'signal': 'BULLISH' if final_pred > 0 else 'BEARISH'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def tool_get_regime_analysis(self) -> Dict[str, Any]:
        """Agent Tool: Get current market regime analysis.

        Returns
        -------
        dict
            Regime analysis with characteristics.
        """
        if not self.is_fitted:
            return {
                'status': 'error',
                'message': 'Model not fitted. Call tool_fit_model first.'
            }

        try:
            current_regime = self.regime_labels[-1]
            regime_mask = self.regime_labels == current_regime
            regime_returns = self.returns[20:][regime_mask]

            return {
                'status': 'success',
                'current_regime': int(current_regime),
                'total_regimes': int(len(np.unique(self.regime_labels))),
                'regime_characteristics': {
                    'mean_return': float(np.mean(regime_returns)),
                    'volatility': float(np.std(regime_returns)),
                    'sample_count': int(len(regime_returns))
                },
                'regime_description': self._describe_regime(regime_returns)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _describe_regime(self, regime_returns: np.ndarray) -> str:
        """Generate human-readable regime description."""
        vol = np.std(regime_returns)
        mean = np.mean(regime_returns)

        if vol > 0.03:
            vol_desc = "high volatility"
        elif vol < 0.01:
            vol_desc = "low volatility"
        else:
            vol_desc = "moderate volatility"

        if mean > 0.001:
            trend_desc = "bullish"
        elif mean < -0.001:
            trend_desc = "bearish"
        else:
            trend_desc = "neutral"

        return f"{trend_desc.capitalize()} market with {vol_desc}"

    def tool_get_trading_signal(self) -> Dict[str, Any]:
        """Agent Tool: Get actionable trading signal.

        Returns
        -------
        dict
            Trading signal with confidence and risk metrics.
        """
        prediction = self.tool_get_prediction()
        regime = self.tool_get_regime_analysis()

        if prediction['status'] == 'error':
            return prediction

        try:
            # Calculate signal strength
            pred_value = prediction['final_prediction']
            volatility = regime['regime_characteristics']['volatility']

            # Signal strength: prediction magnitude relative to volatility
            signal_strength = abs(pred_value) / (volatility + 1e-10)
            confidence = min(signal_strength * 100, 100)  # Cap at 100%

            # Risk assessment
            if volatility > 0.03:
                risk_level = 'HIGH'
                suggested_position = 'REDUCE'
            elif volatility < 0.01:
                risk_level = 'LOW'
                suggested_position = 'NORMAL'
            else:
                risk_level = 'MEDIUM'
                suggested_position = 'NORMAL'

            return {
                'status': 'success',
                'signal': prediction['signal'],
                'confidence': f"{confidence:.1f}%",
                'risk_level': risk_level,
                'suggested_position': suggested_position,
                'prediction': prediction['final_prediction'],
                'regime': regime['regime_description'],
                'recommendation': self._generate_recommendation(
                    prediction['signal'],
                    confidence,
                    risk_level
                )
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _generate_recommendation(
        self,
        signal: str,
        confidence: float,
        risk_level: str
    ) -> str:
        """Generate human-readable trading recommendation."""
        if risk_level == 'HIGH':
            return (
                f"Market volatility is high. Consider reducing position size "
                f"regardless of {signal.lower()} signal."
            )
        elif confidence > 70:
            direction = 'long' if signal == 'BULLISH' else 'short'
            return (
                f"Strong {signal.lower()} signal detected with "
                f"{confidence:.0f}% confidence. Consider {direction} position."
            )
        elif confidence > 40:
            return (
                f"Moderate {signal.lower()} signal. Wait for confirmation "
                f"or use smaller position size."
            )
        else:
            return (
                "Weak signal. No clear direction. Consider staying flat "
                "or hedging."
            )

    def get_all_tools(self) -> Dict[str, callable]:
        """Get all available tools as a dictionary.

        Returns
        -------
        dict
            Dictionary mapping tool names to callable functions.
        """
        return {
            'load_data': self.tool_load_data,
            'fit_model': self.tool_fit_model,
            'get_prediction': self.tool_get_prediction,
            'get_regime_analysis': self.tool_get_regime_analysis,
            'get_trading_signal': self.tool_get_trading_signal
        }


def create_langchain_tools(ticker: str = 'BTC-USD'):
    """Create LangChain-compatible tools from GRM agent tools.

    Parameters
    ----------
    ticker : str
        Asset ticker symbol.

    Returns
    -------
    tuple
        (GRMAgentTools instance, list of tool definitions)

    Example
    -------
    >>> grm_tools, tool_defs = create_langchain_tools('BTC-USD')
    >>> # Use with LangChain agent
    """
    grm_tools = GRMAgentTools(ticker=ticker)

    tool_definitions = [
        {
            'name': 'load_market_data',
            'description': (
                'Load market data for analysis. '
                'Input: JSON with optional start_date (YYYY-MM-DD format).'
            ),
            'func': grm_tools.tool_load_data
        },
        {
            'name': 'fit_grm_model',
            'description': 'Fit the GRM prediction model on loaded data.',
            'func': grm_tools.tool_fit_model
        },
        {
            'name': 'get_prediction',
            'description': 'Get GRM-enhanced price prediction.',
            'func': grm_tools.tool_get_prediction
        },
        {
            'name': 'get_trading_signal',
            'description': (
                'Get actionable trading signal with confidence and risk metrics.'
            ),
            'func': grm_tools.tool_get_trading_signal
        },
        {
            'name': 'get_regime_analysis',
            'description': (
                'Get current market regime analysis and characteristics.'
            ),
            'func': grm_tools.tool_get_regime_analysis
        }
    ]

    return grm_tools, tool_definitions


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("  GRM Agent Tools Demo")
    print("=" * 60)

    tools = GRMAgentTools(ticker='BTC-USD')

    print("\n1. Loading data...")
    result = tools.tool_load_data(start_date='2023-01-01')
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Observations: {result['observations']}")
        print(f"   Date range: {result['date_range']}")
        print(f"   Volatility: {result['volatility']:.4f}")

    print("\n2. Fitting model...")
    result = tools.tool_fit_model()
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Baseline RMSE: {result['baseline_rmse']:.6f}")
        print(f"   Regimes detected: {result['regimes_detected']}")

    print("\n3. Getting prediction...")
    result = tools.tool_get_prediction()
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Signal: {result['signal']}")
        print(f"   Final prediction: {result['final_prediction']:.6f}")
        print(f"   Current alpha: {result['current_alpha']:.3f}")

    print("\n4. Getting regime analysis...")
    result = tools.tool_get_regime_analysis()
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Current regime: {result['current_regime']}")
        print(f"   Description: {result['regime_description']}")

    print("\n5. Getting trading signal...")
    result = tools.tool_get_trading_signal()
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Signal: {result['signal']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Risk level: {result['risk_level']}")
        print(f"   Recommendation: {result['recommendation']}")

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
