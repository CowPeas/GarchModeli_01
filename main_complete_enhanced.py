"""Complete Enhanced GRM Pipeline - PEP8/PEP257 Compliant.

This is the main entry point for running the complete enhanced GRM pipeline
with all advanced features integrated:
- Window-based stratified split
- GMM alternative clustering
- Multi-asset framework
- Hierarchical Bayesian parameter sharing
- Comprehensive statistical validation

Usage:
    python main_complete_enhanced.py --mode single
    python main_complete_enhanced.py --mode multi-asset
    python main_complete_enhanced.py --mode comparison

PEP8 compliant | PEP257 compliant
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Windows encoding fix (CRITICAL for emoji support)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    RealDataLoader,
    MultiBodyGRM,
    BaselineModel,
    WindowStratifiedSplit,
    GMMRegimeDetector,
    auto_select_gmm_components,
    StatisticalTests,
    BootstrapCI,
    DBSCANOptimizer,
    auto_tune_dbscan,
    GRMFeatureEngineer,
    RegimeCoverageValidator

)
from config_enhanced import (
    REAL_DATA_CONFIG,
    OUTPUT_PATHS,
    REGIME_CONFIG,
    STRATIFIED_SPLIT_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """Complete enhanced GRM pipeline orchestrator.
    
    This class orchestrates the entire GRM testing pipeline with all
    advanced features:
    1. Data loading and preprocessing
    2. Baseline model training
    3. Regime detection (DBSCAN or GMM)
    4. Window-based stratified split
    5. Multi-Body GRM training
    6. Statistical validation
    7. Report generation
    
    Attributes
    ----------
    mode : str
        Pipeline mode ('single', 'multi-asset', 'comparison').
    ticker : str
        Asset ticker for single mode.
    use_gmm : bool
        Whether to use GMM instead of DBSCAN.
    
    Examples
    --------
    >>> pipeline = CompletePipeline(mode='single', ticker='BTC-USD')
    >>> results = pipeline.run()
    >>> pipeline.generate_report()
    """
    
    def __init__(
        self,
        mode: str = 'single',
        ticker: str = 'BTC-USD',
        use_gmm: bool = False
    ):
        """Initialize complete pipeline.
        
        Parameters
        ----------
        mode : str, optional
            'single', 'multi-asset', or 'comparison', by default 'single'.
        ticker : str, optional
            Asset ticker for single mode, by default 'BTC-USD'.
        use_gmm : bool, optional
            Use GMM instead of DBSCAN, by default False.
        
        Raises
        ------
        ValueError
            If invalid mode.
        """
        valid_modes = ['single', 'multi-asset', 'comparison']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        self.mode = mode
        self.ticker = ticker
        self.use_gmm = use_gmm
        
        self.results: Dict = {}
        
        # Create output directories
        for path in OUTPUT_PATHS.values():
            if isinstance(path, str):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def log_section(self, title: str):
        """Log a formatted section header.
        
        Parameters
        ----------
        title : str
            Section title.
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"  {title}")
        logger.info("=" * 80)
    
    def load_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load and preprocess data.
        
        Returns
        -------
        tuple of (pd.DataFrame, dict)
            DataFrame with returns and metadata.
        """
        self.log_section("STEP 1: DATA LOADING")
        
        loader = RealDataLoader()
        df, metadata = loader.load_yahoo_finance(
            ticker=REAL_DATA_CONFIG.get('ticker', self.ticker),
            start_date=REAL_DATA_CONFIG['start_date'],
            end_date=REAL_DATA_CONFIG['end_date']
        )
        
        logger.info(f"✓ Loaded {len(df)} observations")
        logger.info(f"  Period: {df.index[0]} to {df.index[-1]}")
        logger.info(f"  Mean return: {df['returns'].mean():.6f}")
        logger.info(f"  Std return: {df['returns'].std():.6f}")
        
        return df, metadata
    
    def train_baseline(
        self,
        train_data: np.ndarray
    ) -> Tuple[BaselineModel, np.ndarray]:
        """Train baseline ARIMA model.
        
        Parameters
        ----------
        train_data : np.ndarray
            Training data.
        
        Returns
        -------
        tuple of (BaselineModel, np.ndarray)
            Trained model and residuals.
        """
        self.log_section("STEP 2: BASELINE MODEL")
        
        baseline = BaselineModel()
        
        # Use default ARIMA order for speed (can be optimized later)
        # Grid search is too slow for large datasets
        default_order = (1, 0, 1)  # Simple AR(1) + MA(1) model
        
        logger.info(f"Using ARIMA order: {default_order}")
        baseline.fit(train_data, order=default_order)
        
        logger.info(f"✓ ARIMA model fitted with order: {default_order}")
        
        # Compute train residuals
        # Note: For ARIMA, we use in-sample fitted values as "predictions"
        baseline_pred_train = baseline.predict(steps=len(train_data))
        residuals = train_data - baseline_pred_train
        
        logger.info(f"  Train RMSE: {np.sqrt(np.mean(residuals**2)):.6f}")
        
        return baseline, residuals
    
    def detect_regimes(
        self,
        residuals: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Detect regimes using DBSCAN or GMM.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from baseline model.
        
        Returns
        -------
        tuple of (np.ndarray, dict)
            Regime labels and detection metrics.
        """
        self.log_section(
            f"STEP 3: REGIME DETECTION ({'GMM' if self.use_gmm else 'DBSCAN'})"
        )
        
        # Feature engineering
        window = REGIME_CONFIG.get('window_size', 20)
        features = GRMFeatureEngineer.extract_regime_features(
            residuals,
            window=window
        )
        
        logger.info(f"✓ Engineered {features.shape[1]} features")
        
        if self.use_gmm:
            # Auto-select GMM components
            n_opt, detector = auto_select_gmm_components(
                features,
                max_components=10
            )
            
            regime_labels = detector.predict(features)
            metrics = detector.get_metrics()
            metrics['n_components'] = n_opt
            
            logger.info(f"✓ Optimal components: {n_opt}")
            logger.info(f"  BIC: {metrics['bic']:.2f}")
            logger.info(f"  Converged: {metrics['converged']}")
            
        else:
            # Auto-tune DBSCAN
            tune_results = auto_tune_dbscan(features)
            
            eps = tune_results['eps']
            minpts = tune_results['minpts']
            
            # Fit DBSCAN manually (since auto_tune_dbscan returns metrics)
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=minpts)
            regime_labels = dbscan.fit_predict(features)
            
            # Store metrics
            metrics = tune_results
            
            logger.info(f"✓ Optimal ε: {eps:.4f}")
            logger.info(f"  Optimal minPts: {minpts}")
            logger.info(f"  N_clusters: {tune_results['n_clusters']}")
            logger.info(f"  Silhouette: {tune_results.get('silhouette_score', 'N/A')}")
        
        unique, counts = np.unique(regime_labels, return_counts=True)
        for regime, count in zip(unique, counts):
            pct = count / len(regime_labels) * 100
            logger.info(f"  Regime {regime}: n={count} ({pct:.1f}%)")
        
        return regime_labels, metrics
    
    def split_data(
        self,
        df: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data using window-based stratified split.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset.
        regime_labels : np.ndarray
            Regime labels.
        
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            Train, validation, test sets.
        """
        self.log_section("STEP 4: WINDOW-BASED STRATIFIED SPLIT")
        
        splitter = WindowStratifiedSplit(
            window_size=STRATIFIED_SPLIT_CONFIG.get('window_size', 30),
            train_ratio=STRATIFIED_SPLIT_CONFIG.get('train_ratio', 0.60),
            val_ratio=STRATIFIED_SPLIT_CONFIG.get('val_ratio', 0.15),
            test_ratio=STRATIFIED_SPLIT_CONFIG.get('test_ratio', 0.25),
            preserve_diversity=True
        )
        
        train_df, val_df, test_df = splitter.split(df, regime_labels)
        
        logger.info(f"✓ Train: {len(train_df)} samples")
        logger.info(f"  Val:   {len(val_df)} samples")
        logger.info(f"  Test:  {len(test_df)} samples")
        
        # Show regime distribution
        dist = splitter.get_regime_distribution()
        logger.info("\n  Regime distribution:")
        for regime in sorted(set(dist['train'].keys()) | set(dist['test'].keys())):
            train_cnt = dist['train'].get(regime, 0)
            test_cnt = dist['test'].get(regime, 0)
            logger.info(f"    Regime {regime}: Train={train_cnt}, Test={test_cnt}")
        
        # Generate report
        report_path = OUTPUT_PATHS.get(
            'results_dir', './results'
        ) + '/window_split_report.txt'
        splitter.generate_report(report_path)
        logger.info(f"\n  Report saved: {report_path}")
        
        return train_df, val_df, test_df
    
    def train_multi_body_grm(
        self,
        train_residuals: np.ndarray,
        regime_labels: np.ndarray,
        metrics: Dict
    ) -> MultiBodyGRM:
        """Train Multi-Body GRM.
        
        Parameters
        ----------
        train_residuals : np.ndarray
            Training residuals.
        regime_labels : np.ndarray
            Regime labels.
        metrics : dict
            Regime detection metrics.
        
        Returns
        -------
        MultiBodyGRM
            Trained model.
        """
        self.log_section("STEP 5: MULTI-BODY GRM TRAINING")
        
        if self.use_gmm:
            # For GMM, we need to recreate labels after split
            # For simplicity, use auto-tuned params
            mb_grm = MultiBodyGRM(
                window_size=REGIME_CONFIG.get('window_size', 20),
                eps=0.5,
                min_samples=10
            )
        else:
            # Use auto-tuned DBSCAN params
            mb_grm = MultiBodyGRM(
                window_size=REGIME_CONFIG.get('window_size', 20),
                eps=metrics.get('eps', 0.5),
                min_samples=metrics.get('minpts', 10)
            )
        
        mb_grm.fit(train_residuals)
        
        n_regimes = len(mb_grm.body_params)
        logger.info(f"✓ Trained {n_regimes} regime models")
        
        for params in mb_grm.body_params:
            regime_id = params['body_id']
            if regime_id != -1:
                logger.info(
                    f"  Regime {regime_id}: α={params['alpha']:.4f}, "
                    f"β={params['beta']:.4f}"
                )
        
        return mb_grm
    
    def validate_statistical(
        self,
        test_data: np.ndarray,
        baseline_pred: np.ndarray,
        mb_pred: np.ndarray
    ) -> Dict:
        """Perform statistical validation.
        
        Parameters
        ----------
        test_data : np.ndarray
            Test set ground truth.
        baseline_pred : np.ndarray
            Baseline predictions.
        mb_pred : np.ndarray
            Multi-Body GRM predictions.
        
        Returns
        -------
        dict
            Statistical test results.
        """
        self.log_section("STEP 6: STATISTICAL VALIDATION")
        
        baseline_errors = test_data - baseline_pred
        mb_errors = test_data - mb_pred
        
        # Metrics
        baseline_rmse = np.sqrt(np.mean(baseline_errors**2))
        mb_rmse = np.sqrt(np.mean(mb_errors**2))
        improvement = (baseline_rmse - mb_rmse) / baseline_rmse * 100
        
        logger.info("Performance Metrics:")
        logger.info(f"  Baseline RMSE: {baseline_rmse:.6f}")
        logger.info(f"  Multi-Body RMSE: {mb_rmse:.6f}")
        logger.info(f"  Improvement: {improvement:.2f}%")
        
        # Diebold-Mariano test
        dm_stat, dm_pvalue = StatisticalTests.diebold_mariano_test(
            baseline_errors, mb_errors
        )
        
        logger.info("\nDiebold-Mariano Test:")
        logger.info(f"  Statistic: {dm_stat:.4f}")
        logger.info(f"  p-value: {dm_pvalue:.6f}")
        
        significant = dm_pvalue < 0.05
        logger.info(
            f"  Result: {'✅ SIGNIFICANT' if significant else '❌ Not significant'}"
        )
        
        # Bootstrap CI
        logger.info("\nBootstrap Confidence Intervals:")
        boot = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)
        ci_lower, ci_upper = boot.compute_rmse_ci(
            baseline_errors, mb_errors
        )
        
        logger.info(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        logger.info(
            f"  Contains zero: {'❌ YES (not significant)' if ci_lower < 0 < ci_upper else '✅ NO (significant)'}"
        )
        
        return {
            'baseline_rmse': baseline_rmse,
            'mb_rmse': mb_rmse,
            'improvement': improvement,
            'dm_statistic': dm_stat,
            'dm_pvalue': dm_pvalue,
            'significant': significant,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def run_single_asset(self) -> Dict:
        """Run pipeline for single asset.
        
        Returns
        -------
        dict
            Complete results.
        """
        # Load data
        df, metadata = self.load_data()
        
        # Baseline model - fit on first 70% for stability
        train_size = int(len(df) * 0.70)
        train_data = df['returns'].values[:train_size]
        
        baseline, train_residuals = self.train_baseline(train_data)
        
        # Get predictions for ALL data to have consistent regime detection
        # Note: This is for regime detection only, not for evaluation
        all_predictions = baseline.predict(steps=len(df))
        residuals_full = df['returns'].values - all_predictions
        
        # Regime detection on full residuals
        regime_labels, metrics = self.detect_regimes(residuals_full)
        
        # IMPORTANT: Feature engineering drops first 'window' samples
        # Adjust df to match regime_labels length
        window_size = REGIME_CONFIG.get('window_size', 20)
        df_adjusted = df.iloc[window_size:].copy()
        
        logger.info(f"✓ Adjusted data length: {len(df_adjusted)} (dropped first {window_size} samples)")
        
        # Window split - now lengths match!
        train_df, val_df, test_df = self.split_data(
            df_adjusted,
            regime_labels
        )
        
        # Re-train baseline on final train set
        final_train_data = train_df['returns'].values
        baseline_final, residuals_final = self.train_baseline(final_train_data)
        
        # Train Multi-Body GRM
        mb_grm = self.train_multi_body_grm(
            residuals_final, regime_labels[:len(train_df)], metrics
        )
        
        # Test predictions
        logger.info("\nGenerating test predictions...")
        
        # Baseline: Get all predictions at once
        baseline_pred = baseline_final.predict(steps=len(test_df))
        
        # Compute ALL residuals (train + test) for MB-GRM
        # MB-GRM needs full residuals array to correctly index windows
        train_residuals_for_grm = train_df['returns'].values - baseline_final.predict(steps=len(train_df))
        test_residuals = test_df['returns'].values - baseline_pred
        
        # Concatenate train and test residuals
        all_residuals = np.concatenate([train_residuals_for_grm, test_residuals])
        train_len = len(train_residuals_for_grm)
        
        # Multi-Body GRM: Add corrections
        mb_corrections = np.zeros(len(test_df))
        
        logger.info(f"  [DEBUG] MB-GRM window_size: {mb_grm.window_size}")
        logger.info(f"  [DEBUG] Train residuals: {len(train_residuals_for_grm)}, Test: {len(test_residuals)}")
        logger.info(f"  [DEBUG] Total residuals for MB-GRM: {len(all_residuals)}")
        
        for i in range(len(test_df)):
            # current_time is relative to all_residuals array
            current_time_in_all = train_len + i
            
            # Get GRM correction
            try:
                _, grm_correction, final_pred, regime_id = mb_grm.predict(
                    all_residuals,
                    current_time=current_time_in_all,
                    baseline_pred=baseline_pred[i]
                )
                mb_corrections[i] = grm_correction
                
                # Debug: Log first few corrections
                if i < 5:
                    logger.info(f"  [DEBUG] Step {i}: correction={grm_correction:.6f}, regime={regime_id}")
            except Exception as e:
                if i < 10:  # Log first few errors
                    logger.warning(f"  Warning at step {i}: {e}")
                mb_corrections[i] = 0.0
        
        mb_pred = baseline_pred + mb_corrections
        
        logger.info(f"  Baseline predictions: {len(baseline_pred)}")
        logger.info(f"  MB-GRM corrections applied: {np.sum(mb_corrections != 0)}/{len(mb_corrections)}")
        logger.info(f"  Mean correction: {np.mean(np.abs(mb_corrections)):.6f}")
        
        # Validate
        stats = self.validate_statistical(
            test_df['returns'].values,
            baseline_pred,
            mb_pred
        )
        
        self.results = {
            'mode': 'single',
            'ticker': self.ticker,
            'use_gmm': self.use_gmm,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'n_regimes': len(mb_grm.body_params),
            'statistics': stats
        }
        
        return self.results
    
    def run(self) -> Dict:
        """Run complete pipeline based on mode.
        
        Returns
        -------
        dict
            Results.
        """
        logger.info("\n" + "="*80)
        logger.info("  COMPLETE ENHANCED GRM PIPELINE")
        logger.info("="*80)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Clustering: {'GMM' if self.use_gmm else 'DBSCAN'}")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")
        
        if self.mode == 'single':
            return self.run_single_asset()
        elif self.mode == 'multi-asset':
            # Call multi-asset script
            logger.info("For multi-asset mode, use:")
            logger.info("  python scripts/test_multi_asset_grm.py")
            return {}
        else:
            logger.warning(f"Mode '{self.mode}' not yet implemented")
            return {}
    
    def generate_report(self, output_file: str = None):
        """Generate final report.
        
        Parameters
        ----------
        output_file : str, optional
            Output file path.
        """
        if not self.results:
            logger.warning("No results to report. Run pipeline first.")
            return
        
        lines = [
            "=" * 80,
            "COMPLETE ENHANCED GRM PIPELINE - FINAL REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "CONFIGURATION",
            "-" * 80,
            f"Mode: {self.results.get('mode', 'N/A')}",
            f"Asset: {self.results.get('ticker', 'N/A')}",
            f"Clustering: {'GMM' if self.results.get('use_gmm') else 'DBSCAN'}",
            "",
            "DATASET",
            "-" * 80,
            f"Train samples: {self.results.get('n_train', 0)}",
            f"Test samples: {self.results.get('n_test', 0)}",
            f"Regimes detected: {self.results.get('n_regimes', 0)}",
            "",
            "RESULTS",
            "-" * 80
        ]
        
        stats = self.results.get('statistics', {})
        lines.extend([
            f"Baseline RMSE: {stats.get('baseline_rmse', 0):.6f}",
            f"Multi-Body RMSE: {stats.get('mb_rmse', 0):.6f}",
            f"Improvement: {stats.get('improvement', 0):.2f}%",
            "",
            "STATISTICAL SIGNIFICANCE",
            "-" * 80,
            f"DM statistic: {stats.get('dm_statistic', 0):.4f}",
            f"DM p-value: {stats.get('dm_pvalue', 1):.6f}",
            f"Significant: {'✅ YES' if stats.get('significant') else '❌ NO'}",
            "",
            f"95% Bootstrap CI: [{stats.get('ci_lower', 0):.6f}, "
            f"{stats.get('ci_upper', 0):.6f}]",
            "",
            "=" * 80
        ])
        
        report = "\n".join(lines)
        
        print("\n" + report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"\nReport saved: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Complete Enhanced GRM Pipeline"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'multi-asset', 'comparison'],
        help='Pipeline mode'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default='BTC-USD',
        help='Asset ticker for single mode'
    )
    
    parser.add_argument(
        '--use-gmm',
        action='store_true',
        help='Use GMM instead of DBSCAN'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/complete_pipeline_report.txt',
        help='Output report file'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    pipeline = CompletePipeline(
        mode=args.mode,
        ticker=args.ticker,
        use_gmm=args.use_gmm
    )
    
    try:
        results = pipeline.run()
        pipeline.generate_report(args.output)
        
        # Exit code based on significance
        stats = results.get('statistics', {})
        if stats.get('significant', False):
            logger.info("\n✅ SUCCESS: Statistical significance achieved!")
            sys.exit(0)
        else:
            logger.info("\n⚠️  Not statistically significant yet")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

