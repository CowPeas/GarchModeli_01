"""
Asset selection ve portfolio optimizasyonu.

Minimum correlation ile maximum diversity sağlayan varlık seçimi.
"""

import numpy as np
from typing import List, Dict


class AssetSelector:
    """Optimal asset portfolio seçici."""
    
    @staticmethod
    def recommended_portfolio() -> Dict[str, Dict]:
        """
        Önceden tanımlanmış optimal portföy.
        
        Returns
        -------
        Dict[str, Dict]
            Varlık bilgileri
        """
        return {
            'BTC-USD': {
                'type': 'cryptocurrency',
                'volatility': 'very_high',
                'regime_dynamics': 'fast',
                'weight': 0.25
            },
            'ETH-USD': {
                'type': 'cryptocurrency',
                'volatility': 'high',
                'regime_dynamics': 'fast',
                'weight': 0.20
            },
            '^GSPC': {
                'type': 'equity_index',
                'volatility': 'medium',
                'regime_dynamics': 'slow',
                'weight': 0.25
            },
            '^VIX': {
                'type': 'volatility_index',
                'volatility': 'very_high',
                'regime_dynamics': 'counter_cyclical',
                'weight': 0.15
            },
            'GC=F': {
                'type': 'commodity',
                'volatility': 'low',
                'regime_dynamics': 'safe_haven',
                'weight': 0.15
            }
        }

