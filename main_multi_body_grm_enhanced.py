"""
Enhanced Multi-Body GRM Test Script - FAZE 6+.

Bu script, Multi-Body GRM modelini gelişmiş özelliklerle eğitir ve test eder.

YENİ ÖZELLİKLER:
- Auto-tuned DBSCAN parameters
- Stratified time series split
- Regime coverage validation
- Comprehensive analysis
"""

import numpy as np
import pandas as pd
import os
import warnings
import sys
from datetime import datetime
from typing import Tuple, Dict

# Windows encoding fix
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Local imports
from models import (
    RealDataLoader,
    BaselineARIMA,
    SchwarzschildGRM,
    MultiBodyGRM,
    calculate_rmse,
    calculate_mae,
    StatisticalTests,
    AdvancedMetrics,
    BootstrapCI,
    RegimeAnalyzer,
    # NEW: Advanced modules
    DBSCANOptimizer,
    auto_tune_dbscan,
    GRMFeatureEngineer,
    StratifiedTimeSeriesSplit,
    RegimeCoverageValidator,
    quick_coverage_check
)

from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    OUTPUT_PATHS,
    SCHWARZSCHILD_CONFIG,
    STATISTICAL_TEST_CONFIG,
    REGIME_CONFIG
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class EnhancedMultiBodyGRMRunner:
    """
    Enhanced Multi-Body GRM test runner.
    
    Bu sınıf, Multi-Body GRM'i gelişmiş özelliklerle çalıştırır:
    - Auto-tuned DBSCAN
    - Stratified split
    - Coverage validation
    """
    
    def __init__(
        self,
        use_stratified_split: bool = True,
        use_auto_tuned_dbscan: bool = True,
        verbose: bool = True
    ):
        """
        EnhancedMultiBodyGRMRunner başlatıcı.
        
        Parameters
        ----------
        use_stratified_split : bool, optional
            Stratified split kullan (varsayılan: True)
        use_auto_tuned_dbscan : bool, optional
            Auto-tuned DBSCAN kullan (varsayılan: True)
        verbose : bool, optional
            Detaylı çıktı (varsayılan: True)
        """
        self.use_stratified_split = use_stratified_split
        self.use_auto_tuned_dbscan = use_auto_tuned_dbscan
        self.verbose = verbose
        
        self.logger = self._setup_logger()
        self.results = {}
    
    def _setup_logger(self):
        """Simple logger setup."""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def log(self, message: str):
        """Log message."""
        if self.verbose:
            print(f"INFO - {message}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Gerçek veri yükle.
        
        Returns
        -------
        pd.DataFrame
            Yüklenen veri
        """
        self.log("\n[ADIM 1] VERİ YÜKLEME")
        self.log("-" * 80)
        
        loader = RealDataLoader()
        
        # Load with yahoo finance
        df, metadata = loader.load_yahoo_finance(
            ticker=REAL_DATA_CONFIG['ticker'],
            start_date=REAL_DATA_CONFIG['start_date'],
            end_date=REAL_DATA_CONFIG['end_date']
        )
        
        # Use returns column (already computed)
        if REAL_DATA_CONFIG.get('use_returns', True):
            df['y'] = df['returns']
        else:
            df['y'] = df['price']
        
        self.log(f"[OK] {len(df)} gözlem yüklendi")
        self.log(f"[OK] Tarih aralığı: {df.index[0]} - {df.index[-1]}")
        
        return df
    
    def split_data_standard(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Standard temporal split.
        
        Parameters
        ----------
        df : pd.DataFrame
            Veri
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train, val, test)
        """
        n = len(df)
        train_end = int(n * SPLIT_CONFIG['train_ratio'])
        val_end = int(n * (SPLIT_CONFIG['train_ratio'] + SPLIT_CONFIG['val_ratio']))
        
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        
        return train, val, test
    
    def split_data_stratified(
        self,
        df: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified split (rejim-aware).
        
        Parameters
        ----------
        df : pd.DataFrame
            Veri
        regime_labels : np.ndarray
            Rejim etiketleri
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train, val, test)
        """
        self.log("\n[STRATIFIED SPLIT] Rejim-aware sampling yapılıyor...")
        
        splitter = StratifiedTimeSeriesSplit(
            train_ratio=SPLIT_CONFIG['train_ratio'],
            val_ratio=SPLIT_CONFIG['val_ratio'],
            test_ratio=SPLIT_CONFIG['test_ratio'],
            preserve_temporal_order=True
        )
        
        train, val, test = splitter.fit_split(df['y'], regime_labels)
        
        # Validate coverage
        is_valid, msg = splitter.validate_coverage()
        self.log(f"[COVERAGE] {msg}")
        
        # Generate report
        report = splitter.generate_report()
        report_file = os.path.join(OUTPUT_PATHS['results_dir'], 'stratified_split_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        self.log(f"[OK] Stratified split raporu: {report_file}")
        
        # Convert back to DataFrame
        train_df = df.loc[train.index]
        val_df = df.loc[val.index]
        test_df = df.loc[test.index]
        
        return train_df, val_df, test_df
    
    def auto_tune_dbscan_params(
        self,
        residuals: np.ndarray,
        window: int = 20
    ) -> Tuple[float, int]:
        """
        DBSCAN parametrelerini otomatik optimize et.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals
        window : int, optional
            Feature window (varsayılan: 20)
            
        Returns
        -------
        Tuple[float, int]
            (optimal_eps, optimal_minpts)
        """
        self.log("\n[AUTO-TUNING] DBSCAN parametreleri optimize ediliyor...")
        
        # Extract features
        features = GRMFeatureEngineer.extract_regime_features(residuals, window)
        features_std, _ = GRMFeatureEngineer.standardize_features(features)
        
        # Auto-tune
        result = auto_tune_dbscan(
            features_std,
            K_desired=5,  # 5 rejim hedefle
            verbose=self.verbose
        )
        
        self.log(f"[OK] Optimal ε: {result['eps']:.4f}")
        self.log(f"[OK] Optimal minPts: {result['minpts']}")
        self.log(f"[OK] Hopkins Statistic: {result['hopkins_statistic']:.4f}")
        self.log(f"[OK] Silhouette Score: {result['silhouette_score']:.4f}")
        
        # Save results
        self.results['dbscan_tuning'] = result
        
        return result['eps'], result['minpts']
    
    def validate_regime_coverage(
        self,
        train_labels: np.ndarray,
        test_labels: np.ndarray
    ):
        """
        Rejim coverage'ını validate et.
        
        Parameters
        ----------
        train_labels : np.ndarray
            Train rejim etiketleri
        test_labels : np.ndarray
            Test rejim etiketleri
        """
        self.log("\n[VALIDATION] Rejim coverage kontrol ediliyor...")
        
        validator = RegimeCoverageValidator(train_labels, test_labels)
        
        # Generate report
        report = validator.generate_report(
            output_file=os.path.join(OUTPUT_PATHS['results_dir'], 'regime_coverage_report.txt')
        )
        
        # Check adequacy
        is_adequate, issues = validator.check_adequacy()
        
        if is_adequate:
            self.log("[OK] ✅ Rejim coverage yeterli!")
        else:
            self.log(f"[WARNING] ⚠️  {len(issues)} coverage sorunu tespit edildi")
            for issue in issues:
                self.log(f"  • {issue}")
            
            # Recommendations
            recommendations = validator.recommend_improvements()
            if recommendations:
                self.log("\n[RECOMMENDATIONS] İyileştirme önerileri:")
                for i, rec in enumerate(recommendations[:3], 1):
                    self.log(f"  {i}. {rec['priority']} {rec['solution']}")
        
        # Save to results
        self.results['coverage_validation'] = {
            'is_adequate': is_adequate,
            'issues': issues,
            'recommendations': validator.recommend_improvements()
        }
    
    def run_complete_pipeline(self):
        """
        Tam pipeline'ı çalıştır.
        
        Returns
        -------
        Dict
            Sonuçlar
        """
        self.log("=" * 80)
        self.log("ENHANCED MULTI-BODY GRM - FULL PIPELINE")
        self.log("=" * 80)
        self.log(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Baseline ARIMA
        self.log("\n[ADIM 2] BASELINE MODEL")
        self.log("-" * 80)
        
        baseline = BaselineARIMA()
        
        # Temporary split for baseline training
        train_temp, val_temp, test_temp = self.split_data_standard(df)
        
        self.log("Baseline ARIMA grid search...")
        best_order = baseline.grid_search(
            train_temp['y'], val_temp['y'],
            p_range=[0, 1, 2],
            d_range=[0, 1],
            q_range=[0, 1, 2],
            verbose=False
        )
        
        baseline.fit(train_temp['y'], order=best_order)
        train_residuals = baseline.get_residuals()
        
        self.log(f"[OK] Baseline: ARIMA{best_order}")
        self.log(f"[OK] Train residuals: {len(train_residuals)}")
        
        # 3. Auto-tune DBSCAN (if enabled)
        if self.use_auto_tuned_dbscan:
            eps, minpts = self.auto_tune_dbscan_params(train_residuals)
        else:
            eps = REGIME_CONFIG.get('dbscan_eps', 0.5)
            minpts = REGIME_CONFIG.get('dbscan_min_samples', 5)
            self.log(f"\n[DBSCAN] Using default params: ε={eps}, minPts={minpts}")
        
        # 4. Multi-Body GRM training
        self.log("\n[ADIM 3] MULTI-BODY GRM EĞİTİMİ")
        self.log("-" * 80)
        
        multi_body_grm = MultiBodyGRM(
            window=SCHWARZSCHILD_CONFIG['window_size'],
            dbscan_eps=eps,
            dbscan_min_samples=minpts
        )
        
        multi_body_grm.fit(train_residuals)
        
        train_regime_labels = multi_body_grm.regime_labels
        n_regimes_train = len(set(train_regime_labels[train_regime_labels != -1]))
        
        self.log(f"[OK] Multi-Body GRM eğitildi!")
        self.log(f"[OK] Train'de {n_regimes_train} rejim tespit edildi")
        
        # 5. Stratified split (if enabled)
        if self.use_stratified_split and n_regimes_train > 1:
            self.log("\n[ADIM 4] STRATIFIED SPLIT")
            self.log("-" * 80)
            
            # Rejim etiketlerini tam veriye extend et
            full_regime_labels = np.full(len(df), -1)
            full_regime_labels[:len(train_regime_labels)] = train_regime_labels
            
            train_df, val_df, test_df = self.split_data_stratified(df, full_regime_labels)
        else:
            self.log("\n[ADIM 4] STANDARD SPLIT")
            self.log("-" * 80)
            train_df, val_df, test_df = self.split_data_standard(df)
        
        self.log(f"[OK] Train: {len(train_df)} gözlem")
        self.log(f"[OK] Val:   {len(val_df)} gözlem")
        self.log(f"[OK] Test:  {len(test_df)} gözlem")
        
        # 6. Re-train on final split
        self.log("\n[ADIM 5] FINAL MODEL EĞİTİMİ")
        self.log("-" * 80)
        
        baseline_final = BaselineARIMA()
        baseline_final.fit(train_df['y'], order=best_order)
        train_residuals_final = baseline_final.get_residuals()
        
        multi_body_grm_final = MultiBodyGRM(
            window=SCHWARZSCHILD_CONFIG['window_size'],
            dbscan_eps=eps,
            dbscan_min_samples=minpts
        )
        
        multi_body_grm_final.fit(train_residuals_final)
        
        # 7. Validate regime coverage
        # Predict test regimes
        test_regime_labels = np.array([
            multi_body_grm_final.predict_regime(test_df['y'].iloc[i:i+1].values)
            for i in range(len(test_df))
        ])
        
        train_regime_labels_final = multi_body_grm_final.regime_labels
        
        self.validate_regime_coverage(train_regime_labels_final, test_regime_labels)
        
        # 8. Testing (simplified - use original script for full testing)
        self.log("\n[ADIM 6] TESTING")
        self.log("-" * 80)
        self.log("[INFO] Test bölümü orijinal script'e bırakıldı")
        self.log("[INFO] Bu enhanced version sadece setup ve validation yapar")
        
        # 9. Summary
        self.log("\n" + "=" * 80)
        self.log("ENHANCED PIPELINE TAMAMLANDI")
        self.log("=" * 80)
        self.log(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'baseline_model': baseline_final,
            'multi_body_grm': multi_body_grm_final,
            'dbscan_params': {'eps': eps, 'minpts': minpts},
            'results': self.results
        }


def main():
    """Ana çalıştırma fonksiyonu."""
    runner = EnhancedMultiBodyGRMRunner(
        use_stratified_split=True,      # Stratified split aktif
        use_auto_tuned_dbscan=True,     # Auto-tuned DBSCAN aktif
        verbose=True
    )
    
    results = runner.run_complete_pipeline()
    
    print("\n" + "=" * 80)
    print("✅ ENHANCED PIPELINE BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)
    print("\nSONRAKİ ADIM:")
    print("  python main.py --multi-body")
    print("  (Full testing için orijinal script'i kullanın)")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import sys
    
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

