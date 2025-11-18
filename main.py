# -*- coding: utf-8 -*-
"""
GRM (Gravitational Residual Model) - Ana Proje Main Dosyası.

Bu dosya, tüm fazları çalıştırmak için merkezi bir kontrol noktası sağlar.
Detaylı loglama, progress tracking ve hata yönetimi içerir.

PEP8 ve PEP257 standartlarına uygun.
"""

import sys
import os
import logging
import warnings
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Windows encoding fix
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Matplotlib backend (GUI olmadan)
import matplotlib
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Local imports
from models import (
    RealDataLoader,
    AlternativeDataLoader,
    BaselineARIMA,
    SchwarzschildGRM,
    KerrGRM
)


class GRMLogger:
    """
    GRM projesi için özelleştirilmiş logger sınıfı.
    
    Hem konsola hem de dosyaya log yazar.
    Renkli çıktı ve progress tracking desteği sağlar.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        verbose: bool = True
    ):
        """
        GRMLogger sınıfını başlatır.
        
        Parameters
        ----------
        log_file : str, optional
            Log dosyası yolu, None ise otomatik oluşturulur
        log_level : int, optional
            Log seviyesi (varsayılan: logging.INFO)
        verbose : bool, optional
            Konsola yazdır (varsayılan: True)
        """
        self.verbose = verbose
        
        # Log dosyası yolu
        if log_file is None:
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'grm_{timestamp}.log'
        
        self.log_file = str(log_file)
        
        # Logger oluştur
        self.logger = logging.getLogger('GRM')
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # Mevcut handler'ları temizle
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_file, encoding='utf-8', mode='w'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (eğer verbose ise)
        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        """Info seviyesinde log yazar."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Warning seviyesinde log yazar."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Error seviyesinde log yazar."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Debug seviyesinde log yazar."""
        self.logger.debug(message)
    
    def section(self, title: str, char: str = '=') -> None:
        """
        Bölüm başlığı yazdırır.
        
        Parameters
        ----------
        title : str
            Bölüm başlığı
        char : str, optional
            Ayırıcı karakter (varsayılan: '=')
        """
        line = char * 80
        self.info(line)
        self.info(f"{title}")
        self.info(line)
    
    def step(self, step_num: int, step_name: str) -> None:
        """
        Adım başlığı yazdırır.
        
        Parameters
        ----------
        step_num : int
            Adım numarası
        step_name : str
            Adım adı
        """
        self.info(f"\n[ADIM {step_num}] {step_name}")
        self.info("-" * 80)
    
    def success(self, message: str) -> None:
        """
        Başarı mesajı yazdırır.
        
        Parameters
        ----------
        message : str
            Mesaj
        """
        self.info(f"[OK] {message}")
    
    def error_msg(self, message: str) -> None:
        """
        Hata mesajı yazdırır.
        
        Parameters
        ----------
        message : str
            Mesaj
        """
        self.error(f"[HATA] {message}")


class GRMMain:
    """
    GRM projesi ana kontrol sınıfı.
    
    Tüm fazları çalıştırmak ve yönetmek için merkezi bir arayüz sağlar.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        verbose: bool = True
    ):
        """
        GRMMain sınıfını başlatır.
        
        Parameters
        ----------
        log_file : str, optional
            Log dosyası yolu
        verbose : bool, optional
            Detaylı çıktı (varsayılan: True)
        """
        self.logger = GRMLogger(log_file=log_file, verbose=verbose)
        self.start_time = datetime.now()
        self.results: Dict[str, Any] = {}
        
        # Dizinleri oluştur
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Gerekli dizinleri oluşturur."""
        directories = [
            'logs',
            'results',
            'visualizations',
            'data',
            'models'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.debug(f"Dizin kontrol edildi: {directory}")
    
    def run_phase(
        self,
        phase_name: str,
        phase_func,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Bir fazı çalıştırır ve sonuçları kaydeder.
        
        Parameters
        ----------
        phase_name : str
            Faz adı
        phase_func : callable
            Çalıştırılacak fonksiyon
        *args
            Fonksiyon argümanları
        **kwargs
            Fonksiyon keyword argümanları
            
        Returns
        -------
        Optional[Any]
            Faz sonuçları
        """
        self.logger.section(f"{phase_name.upper()} BAŞLATILIYOR")
        self.logger.info(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            result = phase_func(*args, **kwargs)
            self.results[phase_name] = {
                'status': 'success',
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            self.logger.success(f"{phase_name} başarıyla tamamlandı!")
            return result
        
        except Exception as e:
            self.logger.error_msg(f"{phase_name} sırasında hata: {str(e)}")
            self.logger.debug(f"Hata detayı: {repr(e)}", exc_info=True)
            self.results[phase_name] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return None
    
    def run_phase1(self) -> Optional[Dict[str, Any]]:
        """
        FAZE 1: Basit Başlangıç simülasyonunu çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Faz sonuçları
        """
        try:
            from main_phase1 import run_phase1_simulation
            return self.run_phase("FAZE 1", run_phase1_simulation)
        except ImportError as e:
            self.logger.error_msg(f"FAZE 1 import hatası: {str(e)}")
            return None
    
    def run_phase2(self) -> Optional[Dict[str, Any]]:
        """
        FAZE 2: Genişletme simülasyonunu çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Faz sonuçları
        """
        try:
            from main_phase2 import run_phase2_simulation
            return self.run_phase("FAZE 2", run_phase2_simulation)
        except ImportError as e:
            self.logger.error_msg(f"FAZE 2 import hatası: {str(e)}")
            return None
    
    def run_phase3(self) -> Optional[Dict[str, Any]]:
        """
        FAZE 3: Gerçek Test simülasyonunu çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Faz sonuçları
        """
        try:
            from main_phase3 import run_phase3_simulation
            return self.run_phase("FAZE 3", run_phase3_simulation)
        except ImportError as e:
            self.logger.error_msg(f"FAZE 3 import hatası: {str(e)}")
            return None
    
    def run_ablation_study(self) -> Optional[Dict[str, Any]]:
        """
        Ablasyon çalışmasını çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Sonuçlar
        """
        try:
            from main_ablation_study import run_ablation_study
            return self.run_phase("ABLATION STUDY", run_ablation_study)
        except ImportError as e:
            self.logger.error_msg(f"Ablasyon çalışması import hatası: {str(e)}")
            return None
    
    def run_cross_validation(self) -> Optional[Dict[str, Any]]:
        """
        Cross-validation çalışmasını çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Sonuçlar
        """
        try:
            from main_cross_validation import run_cross_validation
            return self.run_phase("CROSS VALIDATION", run_cross_validation)
        except ImportError as e:
            self.logger.error_msg(f"Cross-validation import hatası: {str(e)}")
            return None
    
    def run_grn_training(self) -> Optional[Dict[str, Any]]:
        """
        GRN eğitimini çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Sonuçlar
        """
        try:
            from main_grn_train import run_grn_training
            return self.run_phase("GRN TRAINING", run_grn_training)
        except ImportError as e:
            self.logger.error_msg(f"GRN training import hatası: {str(e)}")
            return None
    
    def run_symbolic_discovery(self) -> Optional[Dict[str, Any]]:
        """
        Symbolic regression discovery çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Sonuçlar
        """
        try:
            from main_symbolic_discovery import run_symbolic_discovery
            return self.run_phase("SYMBOLIC DISCOVERY", run_symbolic_discovery)
        except ImportError as e:
            self.logger.error_msg(f"Symbolic discovery import hatası: {str(e)}")
            return None
    
    def run_unified_grm(self) -> Optional[Dict[str, Any]]:
        """
        Unified GRM testini çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Sonuçlar
        """
        try:
            from main_unified_grm import run_unified_grm_test
            return self.run_phase("UNIFIED GRM", run_unified_grm_test)
        except ImportError as e:
            self.logger.error_msg(f"Unified GRM import hatası: {str(e)}")
            return None
    
    def run_multi_body_grm(self) -> Optional[Dict[str, Any]]:
        """
        Multi-Body GRM testini çalıştırır.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Sonuçlar
        """
        try:
            from main_multi_body_grm import run_multi_body_grm_test
            return self.run_phase("MULTI-BODY GRM", run_multi_body_grm_test)
        except ImportError as e:
            self.logger.error_msg(f"Multi-Body GRM import hatası: {str(e)}")
            return None
    
    def print_summary(self) -> None:
        """Tüm sonuçların özetini yazdırır."""
        self.logger.section("ÖZET RAPOR")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info(f"Başlangıç Zamanı: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Bitiş Zamanı: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Toplam Süre: {duration}")
        self.logger.info(f"Log Dosyası: {self.logger.log_file}")
        
        self.logger.info("\nÇALIŞTIRILAN FAZLAR:")
        for phase_name, result in self.results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                self.logger.success(f"  {phase_name}: BAŞARILI")
            else:
                self.logger.error_msg(f"  {phase_name}: HATA - {result.get('error', 'Bilinmeyen hata')}")
        
        self.logger.section("RAPOR TAMAMLANDI")


def main():
    """
    Ana fonksiyon - Komut satırı argümanlarını işler ve fazları çalıştırır.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GRM (Gravitational Residual Model) - Ana Proje Kontrolü',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek Kullanımlar:
  python main.py --phase 1                    # Sadece FAZE 1
  python main.py --phase 3                    # Sadece FAZE 3
  python main.py --all                         # Tüm fazlar
  python main.py --grn                         # GRN eğitimi
  python main.py --multi-body                  # Multi-Body GRM
  python main.py --ablation                    # Ablasyon çalışması
  python main.py --quiet                       # Sessiz mod (sadece dosyaya log)
        """
    )
    
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3],
        help='Çalıştırılacak faz numarası (1, 2, veya 3)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Tüm fazları sırayla çalıştır'
    )
    
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Ablasyon çalışmasını çalıştır'
    )
    
    parser.add_argument(
        '--cross-validation',
        action='store_true',
        help='Cross-validation çalışmasını çalıştır'
    )
    
    parser.add_argument(
        '--grn',
        action='store_true',
        help='GRN eğitimini çalıştır'
    )
    
    parser.add_argument(
        '--symbolic',
        action='store_true',
        help='Symbolic regression discovery çalıştır'
    )
    
    parser.add_argument(
        '--unified',
        action='store_true',
        help='Unified GRM testini çalıştır'
    )
    
    parser.add_argument(
        '--multi-body',
        action='store_true',
        help='Multi-Body GRM testini çalıştır'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Sessiz mod (sadece dosyaya log yaz)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log dosyası yolu (varsayılan: logs/grm_TIMESTAMP.log)'
    )
    
    args = parser.parse_args()
    
    # Main instance oluştur
    main_instance = GRMMain(
        log_file=args.log_file,
        verbose=not args.quiet
    )
    
    # Banner
    main_instance.logger.section("GRM (GRAVITATIONAL RESIDUAL MODEL) PROJESİ")
    main_instance.logger.info("Ana Kontrol Merkezi")
    main_instance.logger.info(f"Python Versiyonu: {sys.version}")
    main_instance.logger.info(f"Çalışma Dizini: {os.getcwd()}")
    
    # Fazları çalıştır
    if args.phase:
        if args.phase == 1:
            main_instance.run_phase1()
        elif args.phase == 2:
            main_instance.run_phase2()
        elif args.phase == 3:
            main_instance.run_phase3()
    
    elif args.all:
        main_instance.logger.info("\nTÜM FAZLAR ÇALIŞTIRILIYOR...\n")
        main_instance.run_phase1()
        main_instance.run_phase2()
        main_instance.run_phase3()
    
    elif args.ablation:
        main_instance.run_ablation_study()
    
    elif args.cross_validation:
        main_instance.run_cross_validation()
    
    elif args.grn:
        main_instance.run_grn_training()
    
    elif args.symbolic:
        main_instance.run_symbolic_discovery()
    
    elif args.unified:
        main_instance.run_unified_grm()
    
    elif args.multi_body:
        main_instance.run_multi_body_grm()
    
    else:
        # Hiçbir argüman verilmediyse, menü göster
        main_instance.logger.info("\nKullanım:")
        main_instance.logger.info("  python main.py --phase 1        # FAZE 1 çalıştır")
        main_instance.logger.info("  python main.py --phase 2        # FAZE 2 çalıştır")
        main_instance.logger.info("  python main.py --phase 3        # FAZE 3 çalıştır")
        main_instance.logger.info("  python main.py --all            # Tüm fazlar")
        main_instance.logger.info("  python main.py --grn            # GRN eğitimi")
        main_instance.logger.info("  python main.py --multi-body     # Multi-Body GRM")
        main_instance.logger.info("  python main.py --ablation       # Ablasyon çalışması")
        main_instance.logger.info("  python main.py --help           # Tüm seçenekler")
        parser.print_help()
    
    # Özet
    main_instance.print_summary()


if __name__ == '__main__':
    main()

