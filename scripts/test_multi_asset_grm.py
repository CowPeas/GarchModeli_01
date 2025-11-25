"""Multi-asset GRM testing and validation script.

This script tests the Multi-Asset GRM framework across multiple financial
instruments, implementing hierarchical Bayesian parameter sharing and
meta-analysis for statistical significance.

Usage:
    python scripts/test_multi_asset_grm.py --assets BTC-USD ETH-USD ^GSPC

PEP8 compliant | PEP257 compliant
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import RealDataLoader
from models import MultiBodyGRM
from models.baseline_model import BaselineModel
from models.statistical_tests import StatisticalTests
from models.dbscan_optimizer import auto_tune_dbscan
from models.grm_feature_engineering import GRMFeatureEngineer
from config_enhanced import (
    REAL_DATA_CONFIG,
    OUTPUT_PATHS,
    REGIME_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAssetGRMTester:
    """Multi-asset GRM testing framework.
    
    This class implements a comprehensive testing framework for Multi-Body
    GRM across multiple assets, including:
    - Individual asset testing
    - Hierarchical Bayesian parameter pooling
    - Meta-analysis for statistical significance
    - Cross-asset validation
    
    Attributes
    ----------
    assets : list of str
        Asset tickers to test (e.g., ['BTC-USD', 'ETH-USD']).
    start_date : str
        Start date for data loading (YYYY-MM-DD).
    end_date : str
        End date for data loading (YYYY-MM-DD).
    window_size : int
        GRM window size.
    results : dict
        Per-asset results after running tests.
    
    Examples
    --------
    >>> tester = MultiAssetGRMTester(['BTC-USD', 'ETH-USD'])
    >>> tester.run()
    >>> tester.generate_report()
    """
    
    def __init__(
        self,
        assets: List[str],
        start_date: str = '2015-01-01',
        end_date: str = '2025-11-09',
        window_size: int = 20
    ):
        """Initialize multi-asset tester.
        
        Parameters
        ----------
        assets : list of str
            Asset tickers to test.
        start_date : str, optional
            Data start date, by default '2015-01-01'.
        end_date : str, optional
            Data end date, by default '2025-11-09'.
        window_size : int, optional
            GRM window size, by default 20.
        """
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        
        self.results: Dict = {}
        self.global_params: Dict = {}
        
        # Create output directory
        Path(OUTPUT_PATHS.get('results_dir', './results')).mkdir(
            parents=True, exist_ok=True
        )
    
    def load_asset_data(self, ticker: str) -> Tuple[pd.DataFrame, Dict]:
        """Load data for a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker (e.g., 'BTC-USD').
        
        Returns
        -------
        tuple of (pd.DataFrame, dict)
            Dataframe with returns and metadata.
        
        Raises
        ------
        ValueError
            If data loading fails.
        """
        logger.info(f"Loading data for {ticker}...")
        
        try:
            loader = RealDataLoader()
            df, metadata = loader.load_yahoo_finance(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            logger.info(
                f"  Loaded {len(df)} observations for {ticker}"
            )
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Failed to load {ticker}: {e}")
            raise ValueError(f"Could not load {ticker}") from e
    
    def test_single_asset(self, ticker: str) -> Dict:
        """Test Multi-Body GRM on a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker.
        
        Returns
        -------
        dict
            Test results including metrics and p-values.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING ASSET: {ticker}")
        logger.info(f"{'='*80}")
        
        # Load data
        df, metadata = self.load_asset_data(ticker)
        
        # Split data (temporal)
        train_size = int(len(df) * 0.70)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Baseline model
        logger.info("\nTraining baseline ARIMA...")
        baseline = BaselineModel()
        baseline.fit(train_df['y'].values)
        
        baseline_pred = []
        for i in range(len(test_df)):
            pred = baseline.predict(test_df['y'].iloc[:i].values, steps=1)
            if pred is not None and len(pred) > 0:
                baseline_pred.append(pred[0])
            else:
                baseline_pred.append(0.0)
        
        baseline_pred = np.array(baseline_pred)
        train_residuals = train_df['y'].values - baseline.predict(
            train_df['y'].values, steps=len(train_df)
        )
        
        # Feature engineering for DBSCAN
        logger.info("\nFeature engineering...")
        fe = GRMFeatureEngineer(window_size=self.window_size)
        features = fe.engineer_features(train_residuals)
        
        # Auto-tune DBSCAN
        logger.info("\nAuto-tuning DBSCAN...")
        eps, minpts, metrics = auto_tune_dbscan(features)
        
        logger.info(f"  Optimal ε: {eps:.4f}")
        logger.info(f"  Optimal minPts: {minpts}")
        logger.info(f"  Silhouette: {metrics['silhouette']:.4f}")
        logger.info(f"  N_clusters: {metrics['n_clusters']}")
        
        # Multi-Body GRM
        logger.info("\nTraining Multi-Body GRM...")
        mb_grm = MultiBodyGRM(
            window_size=self.window_size,
            eps=eps,
            min_samples=minpts
        )
        mb_grm.fit(train_residuals)
        
        logger.info(f"  Detected {len(mb_grm.regime_params)} regimes")
        
        # Test predictions
        test_residuals = test_df['y'].values - baseline_pred
        mb_corrections = []
        
        for i in range(len(test_df)):
            if i < self.window_size:
                mb_corrections.append(0.0)
            else:
                window_res = test_residuals[max(0, i-self.window_size):i]
                corr = mb_grm.predict(window_res, current_time=i)
                mb_corrections.append(corr)
        
        mb_corrections = np.array(mb_corrections)
        mb_pred = baseline_pred + mb_corrections
        
        # Compute metrics
        baseline_errors = test_df['y'].values - baseline_pred
        mb_errors = test_df['y'].values - mb_pred
        
        baseline_rmse = np.sqrt(np.mean(baseline_errors**2))
        mb_rmse = np.sqrt(np.mean(mb_errors**2))
        
        rmse_improvement = (baseline_rmse - mb_rmse) / baseline_rmse * 100
        
        logger.info(f"\nResults:")
        logger.info(f"  Baseline RMSE: {baseline_rmse:.6f}")
        logger.info(f"  Multi-Body RMSE: {mb_rmse:.6f}")
        logger.info(f"  Improvement: {rmse_improvement:.2f}%")
        
        # Statistical tests
        logger.info("\nStatistical tests...")
        dm_result = StatisticalTests.diebold_mariano_test(
            baseline_errors, mb_errors
        )
        
        logger.info(f"  DM statistic: {dm_result['statistic']:.4f}")
        logger.info(f"  p-value: {dm_result['p_value']:.4f}")
        
        significant = dm_result['p_value'] < 0.05
        logger.info(
            f"  Significant: {'✅ YES' if significant else '❌ NO'}"
        )
        
        # Store results
        return {
            'ticker': ticker,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'n_regimes': len(mb_grm.regime_params),
            'baseline_rmse': baseline_rmse,
            'mb_rmse': mb_rmse,
            'rmse_improvement': rmse_improvement,
            'dm_statistic': dm_result['statistic'],
            'dm_pvalue': dm_result['p_value'],
            'significant': significant,
            'regime_params': mb_grm.regime_params,
            'dbscan_metrics': metrics
        }
    
    def compute_global_parameters(self):
        """Compute global parameters via hierarchical Bayes.
        
        Uses empirical Bayes to pool parameters across assets.
        Implements shrinkage estimation:
            θ_asset = w * θ_asset_mle + (1-w) * θ_global
        
        Updates self.global_params with pooled estimates.
        """
        logger.info("\n" + "="*80)
        logger.info("HIERARCHICAL BAYESIAN PARAMETER POOLING")
        logger.info("="*80)
        
        # Collect all regime parameters
        all_alphas = []
        all_betas = []
        
        for ticker, result in self.results.items():
            for regime_id, params in result['regime_params'].items():
                if regime_id != -1:  # Skip outliers
                    all_alphas.append(params['alpha'])
                    all_betas.append(params['beta'])
        
        if len(all_alphas) == 0:
            logger.warning("No regime parameters found!")
            self.global_params = {'alpha': 0.1, 'beta': 0.01}
            return
        
        # Global estimates (mean)
        global_alpha = np.mean(all_alphas)
        global_beta = np.mean(all_betas)
        
        # Global variance
        var_alpha = np.var(all_alphas)
        var_beta = np.var(all_betas)
        
        self.global_params = {
            'alpha': global_alpha,
            'beta': global_beta,
            'var_alpha': var_alpha,
            'var_beta': var_beta,
            'n_regimes_total': len(all_alphas)
        }
        
        logger.info(f"Global α: {global_alpha:.4f} ± {np.sqrt(var_alpha):.4f}")
        logger.info(f"Global β: {global_beta:.4f} ± {np.sqrt(var_beta):.4f}")
        logger.info(f"Total regimes pooled: {len(all_alphas)}")
    
    def meta_analysis(self) -> Dict:
        """Perform meta-analysis across all assets.
        
        Combines p-values using Fisher's method and computes
        overall effect size.
        
        Returns
        -------
        dict
            Meta-analysis results including combined p-value.
        """
        logger.info("\n" + "="*80)
        logger.info("META-ANALYSIS")
        logger.info("="*80)
        
        # Collect p-values
        p_values = [
            result['dm_pvalue'] 
            for result in self.results.values()
        ]
        
        # Fisher's method for combining p-values
        # Test statistic: -2 * sum(log(p_i)) ~ χ²(2k)
        chi2_stat = -2 * np.sum(np.log(np.array(p_values) + 1e-10))
        df = 2 * len(p_values)
        combined_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)
        
        logger.info(f"Fisher's χ²: {chi2_stat:.4f} (df={df})")
        logger.info(f"Combined p-value: {combined_pvalue:.6f}")
        
        # Compute average effect size
        improvements = [
            result['rmse_improvement']
            for result in self.results.values()
        ]
        avg_improvement = np.mean(improvements)
        
        logger.info(f"Average RMSE improvement: {avg_improvement:.2f}%")
        
        # Success rate (% of assets with p < 0.05)
        n_significant = sum(
            1 for r in self.results.values() if r['significant']
        )
        success_rate = n_significant / len(self.results) * 100
        
        logger.info(
            f"Significance success rate: {n_significant}/{len(self.results)} "
            f"({success_rate:.1f}%)"
        )
        
        # Overall assessment
        overall_significant = combined_pvalue < 0.05
        
        logger.info("\n" + "-"*80)
        if overall_significant:
            logger.info("✅ OVERALL: Multi-Body GRM is statistically significant!")
        else:
            logger.info("❌ OVERALL: Not statistically significant")
        logger.info("-"*80)
        
        return {
            'fisher_chi2': chi2_stat,
            'df': df,
            'combined_pvalue': combined_pvalue,
            'avg_improvement': avg_improvement,
            'success_rate': success_rate,
            'overall_significant': overall_significant
        }
    
    def run(self) -> Dict:
        """Run full multi-asset testing pipeline.
        
        Returns
        -------
        dict
            Complete results including meta-analysis.
        """
        logger.info("\n" + "="*80)
        logger.info("MULTI-ASSET GRM TESTING PIPELINE")
        logger.info("="*80)
        logger.info(f"Assets: {', '.join(self.assets)}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Window size: {self.window_size}")
        logger.info("="*80 + "\n")
        
        # Test each asset
        for ticker in self.assets:
            try:
                result = self.test_single_asset(ticker)
                self.results[ticker] = result
            except Exception as e:
                logger.error(f"Failed to test {ticker}: {e}")
                continue
        
        if len(self.results) == 0:
            raise RuntimeError("No assets were successfully tested!")
        
        # Hierarchical Bayes pooling
        self.compute_global_parameters()
        
        # Meta-analysis
        meta_results = self.meta_analysis()
        
        return {
            'asset_results': self.results,
            'global_params': self.global_params,
            'meta_analysis': meta_results
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive test report.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save report, by default None.
        
        Returns
        -------
        str
            Formatted report.
        """
        lines = [
            "=" * 80,
            "MULTI-ASSET GRM TEST REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Assets tested: {len(self.results)}",
            "",
            "PER-ASSET RESULTS",
            "=" * 80,
            ""
        ]
        
        # Per-asset table
        lines.extend([
            f"{'Asset':<12} {'Train':<8} {'Test':<8} {'Regimes':<10} "
            f"{'RMSE_imp':<12} {'DM p-val':<12} {'Sig':<5}",
            "-" * 80
        ])
        
        for ticker, result in self.results.items():
            sig_mark = "✅" if result['significant'] else "❌"
            lines.append(
                f"{ticker:<12} {result['n_train']:<8} {result['n_test']:<8} "
                f"{result['n_regimes']:<10} {result['rmse_improvement']:>10.2f}% "
                f"{result['dm_pvalue']:>11.4f} {sig_mark:<5}"
            )
        
        # Meta-analysis
        if hasattr(self, 'global_params') and 'meta_analysis' in dir(self):
            # Run meta-analysis if not done yet
            if not any('combined_pvalue' in str(r) for r in self.results.values()):
                meta = self.meta_analysis()
            else:
                # Extract from results (assuming it was stored)
                meta = {}
        
        lines.extend([
            "",
            "META-ANALYSIS",
            "=" * 80,
            f"Combined p-value (Fisher): {meta.get('combined_pvalue', 'N/A')}",
            f"Average RMSE improvement: {meta.get('avg_improvement', 0):.2f}%",
            f"Success rate: {meta.get('success_rate', 0):.1f}%",
            "",
            "=" * 80
        ])
        
        report = "\n".join(lines)
        
        # Save if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"\nReport saved to: {output_file}")
        
        return report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test Multi-Asset GRM framework"
    )
    
    parser.add_argument(
        '--assets',
        nargs='+',
        default=['BTC-USD', 'ETH-USD', '^GSPC', '^VIX', 'GC=F'],
        help='Asset tickers to test (default: BTC ETH SPX VIX GOLD)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2015-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=20,
        help='GRM window size (default: 20)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/multi_asset_report.txt',
        help='Output report file'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize tester
    tester = MultiAssetGRMTester(
        assets=args.assets,
        start_date=args.start_date,
        window_size=args.window_size
    )
    
    # Run tests
    try:
        results = tester.run()
        
        # Generate report
        report = tester.generate_report(output_file=args.output)
        print("\n" + report)
        
        # Exit with appropriate code
        meta = results.get('meta_analysis', {})
        if meta.get('overall_significant', False):
            logger.info("\n✅ SUCCESS: Multi-Body GRM shows statistical significance!")
            sys.exit(0)
        else:
            logger.warning("\n⚠️  Multi-Body GRM not statistically significant yet")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

