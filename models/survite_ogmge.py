"""
survite_ogmge.py
================
OGM-GE (On-the-fly Gradient Modulation + Gaussian Enhancement) adapted for
SurvITE multimodal survival analysis.

Original OGM-GE paper: "Balanced Multimodal Learning via On-the-fly Gradient
Modulation" (CVPR 2022).  https://github.com/GeWu-Lab/OGM-GE_CVPR2022

Key idea
--------
In multi-modal learning, one modality often dominates training and the others
remain under-optimised.  OGM-GE corrects this on-the-fly:

  1. **OGM** – after every backward pass, compute a "contribution ratio" k_m
     for each modality m (how strongly it predicts the current batch).
     Rescale each modality encoder's gradients by (1 - k_m) so the dominant
     modality is slowed down and the weak ones catch up.

  2. **GE** – add small Gaussian noise (scaled to the original gradient norm)
     to the rescaled encoder gradients to prevent gradient collapse and
     improve generalisation.

Adaptation to SurvITE
---------------------
SurvITE has a *single* shared encoder that receives a concatenated feature
vector.  To apply OGM-GE per modality we:
  a. Keep modality feature slices separate and give each its own FCNet
     encoder (ModalitySurvITE).
  b. Concatenate their output representations and pass them to the shared
     downstream heads (A=1 / A=0 per time step).
  c. Measure each modality encoder's "contribution" via its mean absolute
     logit contribution in the current batch.
  d. Apply OGM-GE gradient scaling per encoder before optimizer.step().

Imbalance analysis outputs
--------------------------
During training we track:
  • per-modality contribution score (proxy for "how dominant" it is)
  • gradient norms per encoder before / after OGM-GE scaling
  • standard survival metrics: C-index and IBS

These are stored in `history` and can be plotted with the helper functions
at the bottom of this file (compatible with the luad_imbalance_test notebook
style).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ipm_utils import mmd2_lin, wasserstein


# ────────────────────────────────────────────────────────────────────────────
# Helper: small FC network (same as in survite.py)
# ────────────────────────────────────────────────────────────────────────────

class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1, h_dim=100,
                 activation=None, dropout_rate=0.0):
        super().__init__()
        self.activation = activation or nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        curr = in_dim
        if num_layers == 1:
            layers.append(nn.Linear(curr, out_dim))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(curr, h_dim))
                curr = h_dim
            layers.append(nn.Linear(curr, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, dropout_rate=0.0):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = F.dropout(x, p=dropout_rate, training=self.training)
        return self.layers[-1](x)


# ────────────────────────────────────────────────────────────────────────────
# Per-modality dimension projector
# ────────────────────────────────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """
    Squeezes a high-dimensional raw modality (e.g. omics with 2000 features)
    down to a common low-dimensional space (proj_dim, e.g. 32 or 16) before
    it enters the shared-capacity encoder.  This ensures every modality
    arrives at the encoder with the same number of dimensions, preventing
    high-dim modalities from dominating purely by having more input weights.

    Architecture:  Linear(d_in → proj_dim) → BN → activation → Dropout
    A single linear layer is intentional: we want a learned linear
    compression, not a second deep network.  The encoder that follows
    handles the non-linear feature extraction.
    """
    def __init__(self, in_dim: int, proj_dim: int, activation=None):
        super().__init__()
        self.fc  = nn.Linear(in_dim, proj_dim)
        self.bn  = nn.BatchNorm1d(proj_dim)
        self.act = activation or nn.ReLU()

    def forward(self, x, dropout_rate=0.0):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, p=dropout_rate, training=self.training)
        return x


# ────────────────────────────────────────────────────────────────────────────
# OGM-GE utilities
# ────────────────────────────────────────────────────────────────────────────

def _grad_norm(params):
    """Compute total L2 norm of gradients for a list of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def apply_ogm_ge(encoders: dict,
                 contribution_scores: dict,
                 alpha: float = 0.1,
                 apply_ge: bool = True):
    """
    Apply OGM-GE gradient modulation to a dict of modality encoders.

    Parameters
    ----------
    encoders : {name: nn.Module}
        One encoder per modality.
    contribution_scores : {name: float}
        A positive scalar for each modality measuring its current
        "contribution" to the prediction (higher = more dominant).
    alpha : float
        GE noise scale.  Recommended 0.1–0.8 depending on modality gap.
    apply_ge : bool
        Whether to add Gaussian noise (GE part).

    Returns
    -------
    grad_norms_before : {name: float}
    grad_norms_after  : {name: float}
    k_scores          : {name: float}  (the modulation coefficient)
    """
    names = list(encoders.keys())
    scores = np.array([contribution_scores[n] for n in names], dtype=float)

    # Softmax-normalise so scores sum to 1
    scores_exp = np.exp(scores - scores.max())
    scores_norm = scores_exp / (scores_exp.sum() + 1e-9)

    # Modulation coefficient k_m  (eq. 3 in OGM-GE paper):
    #   k_m = 1 - sigmoid(contribution_m - mean_contribution)
    # A modality with above-average contribution gets k < 0.5 → scaled down.
    mean_score = scores_norm.mean()
    k_scores = {n: float(1.0 - torch.sigmoid(
        torch.tensor(scores_norm[i] - mean_score)).item())
        for i, n in enumerate(names)}

    grad_norms_before = {}
    grad_norms_after = {}

    for name, enc in encoders.items():
        params = list(enc.parameters())
        gnorm_before = _grad_norm(params)
        grad_norms_before[name] = gnorm_before

        k = k_scores[name]

        for p in params:
            if p.grad is None:
                continue
            # OGM: scale gradient
            p.grad.data.mul_(k)
            # GE: add Gaussian noise proportional to original norm
            if apply_ge and gnorm_before > 0:
                noise = torch.randn_like(p.grad.data)
                noise_scale = alpha * gnorm_before / (p.grad.data.norm(2) + 1e-9)
                p.grad.data.add_(noise * noise_scale)

        grad_norms_after[name] = _grad_norm(params)

    return grad_norms_before, grad_norms_after, k_scores


# ────────────────────────────────────────────────────────────────────────────
# Multimodal SurvITE with separate per-modality encoders
# ────────────────────────────────────────────────────────────────────────────

class ModalitySurvITE(nn.Module):
    """
    SurvITE variant with *separate* encoders per modality so that OGM-GE
    can modulate gradients independently.

    Parameters
    ----------
    modality_dims : dict  {modality_name: feature_dim}
    t_max : int
    num_Event : int  (kept for API compatibility, currently 1 is supported)
    network_settings : dict  (same schema as original SurvITE)
    device : str
    """

    def __init__(self, modality_dims: dict, t_max: int,
                 network_settings: dict, device='cpu'):
        super().__init__()
        self.device = device
        self.modality_names = list(modality_dims.keys())
        self.t_max = t_max

        z_dim      = network_settings['z_dim']
        h_dim1     = network_settings['h_dim1']
        h_dim2     = network_settings['h_dim2']
        num_layers1 = network_settings['num_layers1']
        num_layers2 = network_settings['num_layers2']
        act_name   = network_settings.get('active_fn', 'relu')

        act = nn.ELU() if act_name == 'elu' else nn.ReLU()

        self.beta      = network_settings.get('beta', 1e-3)
        self.reg_scale = network_settings.get('reg_scale', 0.0)
        self.is_treat  = network_settings.get('is_treat', True)
        self.is_smoothing = network_settings.get('is_smoothing', True)
        self.ipm_term  = network_settings.get('ipm_term', 'no_ipm')
        self.clipping_thres = 10.0

        # ── Per-modality projectors (raw d_m → proj_dim) ──────────────────
        # proj_dim=None disables the projector and keeps the original behaviour.
        self.proj_dim = network_settings.get('proj_dim', None)
        if self.proj_dim is not None:
            self.projectors = nn.ModuleDict({
                name: ModalityProjector(dim, self.proj_dim, act)
                for name, dim in modality_dims.items()
            })
            encoder_in_dim = self.proj_dim
        else:
            self.projectors = None
            # encoder receives raw features directly (original behaviour)
            # each encoder has its own in_dim → handled per-modality below

        # ── Per-modality encoders (proj_dim or d_m → z_dim) ──────────────
        if self.proj_dim is not None:
            # all encoders share the same input dim after projection
            self.encoders = nn.ModuleDict({
                name: FCNet(self.proj_dim, z_dim, num_layers1, h_dim1, act)
                for name in modality_dims.keys()
            })
        else:
            self.encoders = nn.ModuleDict({
                name: FCNet(dim, z_dim, num_layers1, h_dim1, act)
                for name, dim in modality_dims.items()
            })
        # Fusion BN acts on concatenated representation
        self.fusion_bn = nn.BatchNorm1d(z_dim * len(self.modality_names))
        # Optional projection back to z_dim after fusion
        fused_dim = z_dim * len(self.modality_names)
        self.fusion_proj = nn.Linear(fused_dim, z_dim)
        self.fusion_act = act

        # Per-time-step heads (same as original SurvITE)
        self.heads_a1 = nn.ModuleList([
            FCNet(z_dim, 1, num_layers2, h_dim2, act)
            for _ in range(t_max)
        ])
        if self.is_treat:
            self.heads_a0 = nn.ModuleList([
                FCNet(z_dim, 1, num_layers2, h_dim2, act)
                for _ in range(t_max)
            ])
        else:
            self.heads_a0 = None

        self.to(device)

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, x_dict: dict, dropout_rate=0.0):
        """
        Parameters
        ----------
        x_dict : {modality_name: Tensor (B, d_m)}

        Returns
        -------
        modal_reps : {name: Tensor (B, z_dim)}   per-modality encodings
        z          : Tensor (B, z_dim)            fused representation
        logits_a1  : Tensor (B, T)
        logits_a0  : Tensor (B, T)
        """
        modal_reps = {}
        for name, enc in self.encoders.items():
            x = x_dict[name]
            # run projector first if configured
            if self.projectors is not None:
                x = self.projectors[name](x, dropout_rate=dropout_rate)
            h = enc(x, dropout_rate=dropout_rate)
            modal_reps[name] = h

        # Concatenate → BN → project → activate
        concat = torch.cat([modal_reps[n] for n in self.modality_names], dim=1)
        concat = self.fusion_bn(concat)
        z = self.fusion_proj(concat)
        z = self.fusion_act(z)
        z = F.dropout(z, p=dropout_rate, training=self.training)

        logits_a1 = torch.cat([self.heads_a1[m](z) for m in range(self.t_max)], dim=1)
        if self.is_treat:
            logits_a0 = torch.cat([self.heads_a0[m](z) for m in range(self.t_max)], dim=1)
        else:
            logits_a0 = torch.zeros_like(logits_a1)

        return modal_reps, z, logits_a1, logits_a0

    # ── contribution score ─────────────────────────────────────────────────

    @torch.no_grad()
    def contribution_scores(self, modal_reps: dict, logits_a1, logits_a0):
        """
        Proxy contribution of each modality: mean absolute value of the
        representation it contributes to the fused logits, normalised over
        the batch.

        Returns {name: float}
        """
        scores = {}
        for name, rep in modal_reps.items():
            # mean absolute activation of the encoded representation
            scores[name] = rep.abs().mean().item()
        return scores

    # ── survival loss (faithful port of SurvITE.calculate_loss) ───────────

    def calculate_loss(self, x_dict, y, t, a, w, beta, gamma, dropout_rate=0.0):
        modal_reps, z, logits_a1, logits_a0 = self.forward(x_dict, dropout_rate)

        tmp_range = torch.arange(0, self.t_max, device=self.device).float().unsqueeze(0)
        mask1 = (tmp_range == t).float()
        mask2 = (tmp_range <= t).float()
        y_expanded = mask1 * y

        w_clipped = torch.clamp(w, 0., self.clipping_thres)

        # --- IPM Loss (identical logic to SurvITE.calculate_loss) -----------
        loss_ipm = 0.0
        if self.ipm_term != 'no_ipm':
            for m in range(self.t_max):
                idx1 = ((a[:, 0] * mask2[:, m]) == 1).nonzero(as_tuple=False).squeeze()

                idx0 = None
                if self.is_treat:
                    idx0 = (((1. - a[:, 0]) * mask2[:, m]) == 1).nonzero(as_tuple=False).squeeze()

                if idx1.numel() > 0:
                    z_sub = z[idx1]
                    w_sub = w_clipped[idx1, m, 0]
                    if self.ipm_term == 'mmd_lin':
                        loss_ipm += mmd2_lin(z, z_sub, torch.ones_like(z[:, 0]), w_sub)
                    elif self.ipm_term == 'wasserstein':
                        loss_ipm += wasserstein(z, z_sub, torch.ones_like(z[:, 0]), w_sub)

                if self.is_treat and idx0 is not None and idx0.numel() > 0:
                    z_sub = z[idx0]
                    w_sub = w_clipped[idx0, m, 1]
                    if self.ipm_term == 'mmd_lin':
                        loss_ipm += mmd2_lin(z, z_sub, torch.ones_like(z[:, 0]), w_sub)
                    elif self.ipm_term == 'wasserstein':
                        loss_ipm += wasserstein(z, z_sub, torch.ones_like(z[:, 0]), w_sub)

        # --- Smoothing Loss --------------------------------------------------
        loss_smoothing = 0.0
        if self.is_smoothing:
            for m in range(1, self.t_max):
                for pc, pp in zip(self.heads_a1[m].parameters(),
                                  self.heads_a1[m - 1].parameters()):
                    loss_smoothing += torch.mean((pc - pp) ** 2)
                if self.is_treat:
                    for pc, pp in zip(self.heads_a0[m].parameters(),
                                      self.heads_a0[m - 1].parameters()):
                        loss_smoothing += torch.mean((pc - pp) ** 2)

        # --- Factual Loss (weighted BCE) ------------------------------------
        denom1 = torch.sum(mask2 * a * w[:, :, 0], dim=0, keepdim=True) + 1e-8
        tmp_w1 = (w[:, :, 0] / denom1) * mask2 * a
        bce_a1 = F.binary_cross_entropy_with_logits(logits_a1, y_expanded, reduction='none')
        loss_factual = torch.sum(tmp_w1 * bce_a1)

        if self.is_treat:
            denom0 = torch.sum(mask2 * (1. - a) * w[:, :, 1], dim=0, keepdim=True) + 1e-8
            tmp_w0 = (w[:, :, 1] / denom0) * mask2 * (1. - a)
            bce_a0 = F.binary_cross_entropy_with_logits(logits_a0, y_expanded, reduction='none')
            loss_factual += torch.sum(tmp_w0 * bce_a0)

        # --- L2 Regularisation (encoders only) ------------------------------
        loss_reg = 0.0
        if self.reg_scale > 0:
            for enc in self.encoders.values():
                for p in enc.parameters():
                    loss_reg += torch.sum(p ** 2)
            loss_reg *= self.reg_scale

        total = loss_factual + beta * loss_ipm + gamma * loss_smoothing + loss_reg
        return total, loss_factual, loss_ipm, modal_reps, logits_a1, logits_a0

    # ── training step with OGM-GE ──────────────────────────────────────────

    def train_step_ogmge(self, optimizer, x_dict, y, t, a, w,
                         beta=1e-3, gamma=1e-3, dropout_rate=0.0,
                         alpha=0.1, apply_ge=True,
                         modulation_active=True):
        """
        Full OGM-GE training step.

        Returns
        -------
        loss_total, loss_fact,
        contrib_scores  : {name: float}
        grad_norms_before : {name: float}
        grad_norms_after  : {name: float}
        k_scores          : {name: float}
        """
        self.train()
        optimizer.zero_grad()

        # Move inputs to device
        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32).to(self.device)

        x_d = {k: _t(v) for k, v in x_dict.items()}
        y, t, a, w = _t(y), _t(t), _t(a), _t(w)

        gamma_eff = gamma if self.is_smoothing else 0.0

        loss_total, loss_fact, loss_ipm, modal_reps, logits_a1, logits_a0 = \
            self.calculate_loss(x_d, y, t, a, w, beta, gamma_eff, dropout_rate)

        loss_total.backward()

        # Contribution scores from current forward pass
        contrib = self.contribution_scores(modal_reps, logits_a1, logits_a0)

        grad_before, grad_after, k_scores = {}, {}, {}
        if modulation_active and len(self.modality_names) > 1:
            # Build a combined module per modality: projector (if any) + encoder.
            # OGM-GE should see and modulate the full per-modality parameter set.
            combined = {}
            for n in self.modality_names:
                combined[n] = nn.ModuleList(
                    [self.projectors[n], self.encoders[n]]
                    if self.projectors is not None
                    else [self.encoders[n]]
                )
            grad_before, grad_after, k_scores = apply_ogm_ge(
                combined, contrib, alpha=alpha, apply_ge=apply_ge)

        optimizer.step()

        return (loss_total.item(), loss_fact.item(), loss_ipm if isinstance(loss_ipm, float) else loss_ipm.item(),
                contrib, grad_before, grad_after, k_scores)

    def train_baseline_ogmge(self, optimizer, x_dict, y, t, a,
                             dropout_rate=0.0, alpha=0.1, apply_ge=True,
                             modulation_active=True):
        """Convenience wrapper with unit weights (mirrors train_baseline)."""
        n = list(x_dict.values())[0].shape[0]
        w = np.ones([n, self.t_max, 2])
        return self.train_step_ogmge(
            optimizer, x_dict, y, t, a, w,
            beta=self.beta, gamma=0.0,
            dropout_rate=dropout_rate,
            alpha=alpha, apply_ge=apply_ge,
            modulation_active=modulation_active
        )

    # ── inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_hazard(self, x_dict, is_treat_group=True):
        self.eval()
        x_d = {k: torch.tensor(v, dtype=torch.float32).to(self.device)
               for k, v in x_dict.items()}
        _, _, logits_a1, logits_a0 = self.forward(x_d, dropout_rate=0.0)
        logits = logits_a1 if is_treat_group else logits_a0
        odd = torch.exp(logits)
        return (odd / (1. + odd)).cpu().numpy()

    def predict_survival_A1(self, x_dict):
        return np.cumprod(1. - self._predict_hazard(x_dict, True), axis=1)

    def predict_survival_A0(self, x_dict):
        return np.cumprod(1. - self._predict_hazard(x_dict, False), axis=1)
