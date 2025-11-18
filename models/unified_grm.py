"""
Unified End-to-End GRM Model - LSTM Baseline + GRN Correction.

Bu modül, baseline tahmin ve rezidüel düzeltmesini birlikte optimize eder.

FAZE 6: PIML İLERİ SEVİYE
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Local imports
from models.grn_network import GravitationalResidualNetwork


class UnifiedGRM(nn.Module):
    """
    Unified End-to-End GRM Model.
    
    Bu model, baseline tahmin (LSTM) ve rezidüel düzeltmesini (GRN)
    birlikte optimize eder. İki bileşen ayrı ayrı değil, birlikte
    öğrenilir.
    
    Architecture:
        Input: x_history (batch, seq_len, features)
          ↓
        [LSTM Baseline]
          ↓
        baseline_pred
          ↓
        [Compute Residuals]
          ↓
        [GRN Correction]
          ↓
        final_pred = baseline_pred + grm_correction
    
    Attributes
    ----------
    lstm : nn.LSTM
        Baseline LSTM modeli
    lstm_output : nn.Linear
        LSTM çıktısını tahmine dönüştüren linear layer
    grn : GravitationalResidualNetwork
        Rezidüel düzeltmesi için GRN
    """
    
    def __init__(
        self,
        input_size: int = 1,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        grn_hidden_sizes: List[int] = [64, 32, 16],
        use_monotonicity: bool = True,
        use_energy_conservation: bool = True
    ):
        """
        UnifiedGRM sınıfını başlatır.
        
        Parameters
        ----------
        input_size : int, optional
            Girdi feature sayısı (varsayılan: 1)
        lstm_hidden_size : int, optional
            LSTM gizli katman boyutu (varsayılan: 64)
        lstm_num_layers : int, optional
            LSTM katman sayısı (varsayılan: 2)
        grn_hidden_sizes : List[int], optional
            GRN gizli katman boyutları (varsayılan: [64, 32, 16])
        use_monotonicity : bool, optional
            GRN'de monotoniklik kısıtlaması (varsayılan: True)
        use_energy_conservation : bool, optional
            GRN'de enerji korunumu kısıtlaması (varsayılan: True)
        """
        super().__init__()
        
        # Baseline LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.2 if lstm_num_layers > 1 else 0.0
        )
        self.lstm_output = nn.Linear(lstm_hidden_size, 1)
        
        # GRN for residual correction
        self.grn = GravitationalResidualNetwork(
            input_size=4,  # M, a, τ, ε
            hidden_sizes=grn_hidden_sizes,
            output_size=1,
            use_monotonicity=use_monotonicity,
            use_energy_conservation=use_energy_conservation
        )
    
    def compute_autocorr(
        self,
        residuals: torch.Tensor,
        lag: int = 1
    ) -> torch.Tensor:
        """
        Otokorelasyon hesapla (batch için).
        
        Parameters
        ----------
        residuals : torch.Tensor, shape (batch, seq_len)
            Artık dizisi
        lag : int, optional
            Lag değeri (varsayılan: 1)
            
        Returns
        -------
        torch.Tensor, shape (batch, 1)
            Otokorelasyon değerleri
        """
        batch_size, seq_len = residuals.shape
        
        if seq_len <= lag:
            return torch.zeros(batch_size, 1, device=residuals.device)
        
        # Lag'li seriler
        x_t = residuals[:, :-lag]
        x_t_lag = residuals[:, lag:]
        
        # Mean center
        x_t_mean = x_t.mean(dim=1, keepdim=True)
        x_t_lag_mean = x_t_lag.mean(dim=1, keepdim=True)
        
        x_t_centered = x_t - x_t_mean
        x_t_lag_centered = x_t_lag - x_t_lag_mean
        
        # Correlation
        numerator = (x_t_centered * x_t_lag_centered).sum(dim=1, keepdim=True)
        denominator = torch.sqrt(
            (x_t_centered ** 2).sum(dim=1, keepdim=True) *
            (x_t_lag_centered ** 2).sum(dim=1, keepdim=True)
        ) + 1e-8
        
        autocorr = numerator / denominator
        
        # Clip to [-1, 1]
        autocorr = torch.clamp(autocorr, -1.0, 1.0)
        
        return autocorr
    
    def forward(
        self,
        x_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x_history : torch.Tensor, shape (batch, seq_len, features)
            Geçmiş zaman serisi verisi
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (baseline_pred, grm_correction, final_pred)
        """
        batch_size, seq_len, features = x_history.shape
        
        # Baseline LSTM prediction
        lstm_out, _ = self.lstm(x_history)
        baseline_pred = self.lstm_output(lstm_out[:, -1, :])  # (batch, 1)
        
        # Compute residuals (detach to prevent gradient flow)
        # Residuals = actual - baseline_pred
        # For training, we use x_history as "actual"
        residuals = x_history[:, :, 0] - baseline_pred.detach()  # (batch, seq_len)
        
        # Compute GRM features
        # Mass (variance)
        mass = torch.var(residuals, dim=1, keepdim=True)  # (batch, 1)
        
        # Spin (autocorrelation)
        spin = self.compute_autocorr(residuals)  # (batch, 1)
        
        # Tau (time since shock) - simplified for now
        tau = torch.ones_like(mass) * 5.0  # (batch, 1)
        
        # GRM correction
        grm_correction = self.grn(
            mass=mass,
            spin=spin,
            tau=tau,
            residuals_history=residuals
        )  # (batch, 1)
        
        # Final prediction
        final_pred = baseline_pred + grm_correction
        
        return baseline_pred, grm_correction, final_pred
    
    def combined_loss(
        self,
        baseline_pred: torch.Tensor,
        grm_correction: torch.Tensor,
        final_pred: torch.Tensor,
        targets: torch.Tensor,
        mass: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combined loss: Final + Baseline + Physics.
        
        L_total = L_final + λ_baseline * L_baseline + λ_physics * L_physics
        
        Parameters
        ----------
        baseline_pred : torch.Tensor
            Baseline tahminleri
        grm_correction : torch.Tensor
            GRM düzeltmeleri
        final_pred : torch.Tensor
            Final tahminler
        targets : torch.Tensor
            Hedef değerler
        mass : torch.Tensor
            Kütle değerleri (physics loss için)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (total_loss, loss_final, loss_baseline, loss_physics)
        """
        # Final prediction loss
        loss_final = nn.MSELoss()(final_pred, targets)
        
        # Baseline loss (encourages baseline to be good on its own)
        loss_baseline = nn.MSELoss()(baseline_pred, targets)
        
        # Physics loss (from GRN)
        loss_physics = self.grn.physics_loss(mass, grm_correction)
        
        # Combined loss
        total_loss = loss_final + 0.1 * loss_baseline + 0.05 * loss_physics
        
        return total_loss, loss_final, loss_baseline, loss_physics
    
    def predict(
        self,
        x_history: np.ndarray,
        device: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tahmin yap (numpy input/output).
        
        Parameters
        ----------
        x_history : np.ndarray, shape (seq_len, features) or (batch, seq_len, features)
            Geçmiş zaman serisi verisi
        device : str, optional
            Cihaz ('cpu' veya 'cuda'), None ise model'in cihazını kullanır
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (baseline_pred, grm_correction, final_pred)
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Input hazırlama
        if x_history.ndim == 2:
            x_history = x_history[np.newaxis, :, :]  # (1, seq_len, features)
        
        x_tensor = torch.FloatTensor(x_history).to(device)
        
        with torch.no_grad():
            baseline_pred, grm_correction, final_pred = self.forward(x_tensor)
        
        # Numpy'a çevir
        baseline_pred = baseline_pred.cpu().numpy()
        grm_correction = grm_correction.cpu().numpy()
        final_pred = final_pred.cpu().numpy()
        
        # Batch dimension'ı kaldır
        if baseline_pred.shape[0] == 1:
            baseline_pred = baseline_pred[0, 0]
            grm_correction = grm_correction[0, 0]
            final_pred = final_pred[0, 0]
        
        return baseline_pred, grm_correction, final_pred

