"""
GRN Trainer - Eğitim Modülü.

Bu modül, GRN modelinin eğitim sürecini yönetir.

FAZE 5: PIML TEMEL ENTEGRASYONU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class GRMDataSet(Dataset):
    """
    GRM veri seti sınıfı.
    
    PyTorch DataLoader ile uyumlu veri seti.
    """
    
    def __init__(
        self,
        mass: np.ndarray,
        spin: np.ndarray,
        tau: np.ndarray,
        residuals_history: np.ndarray,
        targets: np.ndarray
    ):
        """
        GRMDataSet sınıfını başlatır.
        
        Parameters
        ----------
        mass : np.ndarray
            Kütle dizisi
        spin : np.ndarray
            Dönme dizisi
        tau : np.ndarray
            Time since shock dizisi
        residuals_history : np.ndarray
            Geçmiş artıklar dizisi
        targets : np.ndarray
            Hedef değerler (gelecekteki artık)
        """
        self.mass = torch.FloatTensor(mass).unsqueeze(1)  # (n, 1)
        self.spin = torch.FloatTensor(spin).unsqueeze(1)  # (n, 1)
        self.tau = torch.FloatTensor(tau).unsqueeze(1)  # (n, 1)
        self.residuals_history = torch.FloatTensor(residuals_history)  # (n, window_size)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)  # (n, 1)
    
    def __len__(self) -> int:
        """
        Veri seti boyutunu döndürür.
        
        Returns
        -------
        int
            Veri seti boyutu
        """
        return len(self.mass)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        İndeks ile veri öğesi alır.
        
        Parameters
        ----------
        idx : int
            İndeks
            
        Returns
        -------
        tuple
            (mass, spin, tau, residuals_history, targets)
        """
        return (
            self.mass[idx],
            self.spin[idx],
            self.tau[idx],
            self.residuals_history[idx],
            self.targets[idx]
        )


class GRNTrainer:
    """
    GRN model trainer with physics-informed loss.
    
    Bu sınıf, GRN modelinin eğitim sürecini yönetir.
    Early stopping, model kaydetme gibi özellikler içerir.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        GRNTrainer sınıfını başlatır.
        
        Parameters
        ----------
        model : nn.Module
            GRN modeli
        learning_rate : float, optional
            Öğrenme hızı (varsayılan: 0.001)
        device : str, optional
            Cihaz ('cpu' veya 'cuda'), None ise otomatik seçilir
        """
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'data_loss': [],
            'physics_loss': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Bir epoch için eğitim yapar.
        
        Parameters
        ----------
        train_loader : DataLoader
            Eğitim veri yükleyici
            
        Returns
        -------
        float
            Ortalama eğitim kaybı
        """
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            mass, spin, tau, residuals_history, targets = batch
            
            # Cihaza taşı
            mass = mass.to(self.device)
            spin = spin.to(self.device)
            tau = tau.to(self.device)
            residuals_history = residuals_history.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(mass, spin, tau, residuals_history)
            
            # Loss
            loss, data_loss, physics_loss = self.model.combined_loss(
                predictions, targets, mass, predictions
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_data_loss = total_data_loss / n_batches if n_batches > 0 else 0.0
        avg_physics_loss = total_physics_loss / n_batches if n_batches > 0 else 0.0
        
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['data_loss'].append(avg_data_loss)
        self.training_history['physics_loss'].append(avg_physics_loss)
        
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Validation seti üzerinde değerlendirme yapar.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation veri yükleyici
            
        Returns
        -------
        float
            Ortalama validation kaybı
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                mass, spin, tau, residuals_history, targets = batch
                
                # Cihaza taşı
                mass = mass.to(self.device)
                spin = spin.to(self.device)
                tau = tau.to(self.device)
                residuals_history = residuals_history.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(mass, spin, tau, residuals_history)
                
                # Loss
                loss, _, _ = self.model.combined_loss(
                    predictions, targets, mass, predictions
                )
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        self.training_history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping: int = 10,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Full training loop with early stopping.
        
        Parameters
        ----------
        train_loader : DataLoader
            Eğitim veri yükleyici
        val_loader : DataLoader
            Validation veri yükleyici
        epochs : int, optional
            Maksimum epoch sayısı (varsayılan: 100)
        early_stopping : int, optional
            Early stopping patience (varsayılan: 10)
        save_path : str, optional
            Model kayıt yolu, None ise 'models/grn_best.pth' kullanılır
            
        Returns
        -------
        dict
            Eğitim geçmişi
        """
        if save_path is None:
            save_path = 'models/grn_best.pth'
        
        # Dizin oluştur
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n[GRN] Eğitim başlatılıyor...")
        print("-" * 80)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), save_path)
                print(f"  → Yeni en iyi model kaydedildi (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"\n[GRN] Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(save_path))
        print(f"\n[GRN] En iyi model yüklendi (Val Loss: {best_val_loss:.6f})")
        print("-" * 80)
        
        return self.training_history

