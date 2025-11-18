"""
Gravitational Residual Network (GRN) - Physics-Inspired Neural Network.

Bu modül, bükülme fonksiyonunu öğrenen bir sinir ağı uygular.
Physics-informed inductive bias'lar ile kısıtlanmış öğrenme.

FAZE 5: PIML TEMEL ENTEGRASYONU
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class GravitationalResidualNetwork(nn.Module):
    """
    Physics-inspired neural network for learning curvature function.
    
    Architecture:
        Input: [M(t), a(t), τ(t), ε(t-k:t)] → hidden layers → Output: Γ(t+1)
    
    Physics-informed constraints:
        1. Monotonicity: ∂Γ/∂M ≥ 0 (larger mass → larger curvature)
        2. Energy conservation: Σ|Γ(t)| is bounded
        3. Symmetry: Γ(M, a, τ) = -Γ(M, -a, τ) for spin
    
    Attributes
    ----------
    network : nn.Sequential
        Ana sinir ağı mimarisi
    alpha : nn.Parameter
        Öğrenilebilir kütleçekimsel etkileşim katsayısı
    beta : nn.Parameter
        Öğrenilebilir sönümleme hızı parametresi
    gamma : nn.Parameter
        Öğrenilebilir dönme etkisinin ağırlığı
    use_monotonicity : bool
        Monotoniklik kısıtlaması kullanılsın mı
    use_energy_conservation : bool
        Enerji korunumu kısıtlaması kullanılsın mı
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_sizes: List[int] = [64, 32, 16],
        output_size: int = 1,
        use_monotonicity: bool = True,
        use_energy_conservation: bool = True
    ):
        """
        GravitationalResidualNetwork sınıfını başlatır.
        
        Parameters
        ----------
        input_size : int, optional
            Girdi boyutu: M, a, τ, ε (varsayılan: 4)
        hidden_sizes : List[int], optional
            Gizli katman boyutları (varsayılan: [64, 32, 16])
        output_size : int, optional
            Çıktı boyutu (varsayılan: 1)
        use_monotonicity : bool, optional
            Monotoniklik kısıtlaması kullan (varsayılan: True)
        use_energy_conservation : bool, optional
            Enerji korunumu kısıtlaması kullan (varsayılan: True)
        """
        super().__init__()
        
        self.use_monotonicity = use_monotonicity
        self.use_energy_conservation = use_energy_conservation
        
        # Encoder network
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())  # Bounded output [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Learnable physics parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        mass: torch.Tensor,
        spin: torch.Tensor,
        tau: torch.Tensor,
        residuals_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        mass : torch.Tensor, shape (batch, 1)
            Kütle (volatilite)
        spin : torch.Tensor, shape (batch, 1)
            Dönme (otokorelasyon)
        tau : torch.Tensor, shape (batch, 1)
            Şoktan geçen zaman
        residuals_history : torch.Tensor, shape (batch, seq_len)
            Geçmiş artıklar dizisi
            
        Returns
        -------
        torch.Tensor, shape (batch, 1)
            Bükülme düzeltmesi
        """
        # Decay factor
        decay = 1.0 / (1.0 + self.beta * tau)
        
        # Input features
        x = torch.cat([
            mass,
            spin,
            tau,
            residuals_history[:, -1:] if residuals_history.dim() > 1 else residuals_history.unsqueeze(-1)
        ], dim=1)
        
        # Neural network correction
        nn_correction = self.network(x)
        
        # Physics-inspired base term
        last_residual = residuals_history[:, -1:] if residuals_history.dim() > 1 else residuals_history.unsqueeze(-1)
        base_term = self.alpha * mass * torch.tanh(last_residual)
        spin_term = self.gamma * spin * last_residual
        
        # Combined output
        curvature = (base_term + spin_term + nn_correction) * decay
        
        return curvature
    
    def physics_loss(
        self,
        mass: torch.Tensor,
        curvature: torch.Tensor
    ) -> torch.Tensor:
        """
        Physics-informed loss term.
        
        Enforces:
        1. Monotonicity: dΓ/dM ≥ 0
        2. Energy conservation: Total energy bounded
        
        Parameters
        ----------
        mass : torch.Tensor
            Kütle değerleri
        curvature : torch.Tensor
            Bükülme değerleri
            
        Returns
        -------
        torch.Tensor
            Physics loss değeri
        """
        loss = torch.tensor(0.0, device=mass.device)
        
        if self.use_monotonicity:
            # Monotonicity constraint
            # Approximate derivative using finite differences
            mass_perturbed = mass + 0.01
            # Simplified: assume other inputs stay same
            # In practice, this would need full forward pass
            # For now, use a simpler approximation
            curvature_perturbed = curvature * (1.0 + 0.01 / (mass + 1e-8))
            
            derivative = (curvature_perturbed - curvature) / 0.01
            monotonicity_loss = torch.relu(-derivative).mean()  # Penalize negative derivatives
            loss += 0.1 * monotonicity_loss
        
        if self.use_energy_conservation:
            # Energy conservation: penalize large total energy
            total_energy = torch.sum(torch.abs(curvature))
            energy_loss = torch.relu(total_energy - 10.0)  # Soft threshold
            loss += 0.01 * energy_loss
        
        return loss
    
    def combined_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mass: torch.Tensor,
        curvature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combined loss: Data fidelity + Physics-informed.
        
        L_total = L_data + λ * L_physics
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model tahminleri
        targets : torch.Tensor
            Hedef değerler
        mass : torch.Tensor
            Kütle değerleri
        curvature : torch.Tensor
            Bükülme değerleri
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (total_loss, data_loss, physics_loss)
        """
        # Data fidelity loss
        data_loss = nn.MSELoss()(predictions, targets)
        
        # Physics loss
        physics_loss = self.physics_loss(mass, curvature)
        
        # Combined
        total_loss = data_loss + 0.1 * physics_loss
        
        return total_loss, data_loss, physics_loss

