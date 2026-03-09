import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.ipm_utils import mmd2_lin, wasserstein


# --- Helper Network ---
class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1, h_dim=100, activation=nn.ReLU(), dropout_rate=0.0):
        super(FCNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
        curr_dim = in_dim
        if num_layers == 1:
            self.layers.append(nn.Linear(curr_dim, out_dim))
        else:
            for i in range(num_layers - 1):
                self.layers.append(nn.Linear(curr_dim, h_dim))
                curr_dim = h_dim
            self.layers.append(nn.Linear(curr_dim, out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        # Last layer (linear output usually, activation handled externally if needed)
        x = self.layers[-1](x)
        return x

# --- Main Class ---
class SurvITE(nn.Module):
    def __init__(self, input_dims, network_settings, device='cpu'):
        super(SurvITE, self).__init__()
        self.device = device

        self.x_dim = input_dims['x_dim']
        self.t_max = input_dims['t_max']
        self.num_Event = input_dims['num_Event']

        self.z_dim = network_settings['z_dim']
        self.h_dim1 = network_settings['h_dim1']
        self.h_dim2 = network_settings['h_dim2']
        self.num_layers1 = network_settings['num_layers1']
        self.num_layers2 = network_settings['num_layers2']

        # Activation map
        if network_settings['active_fn'] == 'relu':
            self.active_fn = nn.ReLU()
        elif network_settings['active_fn'] == 'elu':
            self.active_fn = nn.ELU()
        else:
            self.active_fn = nn.ReLU()

        self.beta = network_settings['beta']
        self.reg_scale = network_settings.get('reg_scale', 0.0)
        self.ipm_term = network_settings['ipm_term']
        self.is_treat = network_settings['is_treat']
        self.is_smoothing = network_settings['is_smoothing']
        self.clipping_thres = 10.0

        self._build_net()
        self.to(self.device)

    def _build_net(self):
        # 1. Encoder PHI(x)
        self.encoder = FCNet(self.x_dim, self.z_dim, self.num_layers1, self.h_dim1, 
                             self.active_fn, dropout_rate=0.0) # Dropout set dynamically during forward

        # TF code does Batch Norm AFTER encoder before splitting
        self.bn = nn.BatchNorm1d(self.z_dim)

        # 2. Hypothesis Layers H(Z; A, T)
        # We need a separate network for each time step m in 0...t_max-1
        # For A=1
        self.heads_a1 = nn.ModuleList([
            FCNet(self.z_dim, 1, self.num_layers2, self.h_dim2, self.active_fn)
            for _ in range(self.t_max)
        ])

        # For A=0
        if self.is_treat:
            self.heads_a0 = nn.ModuleList([
                FCNet(self.z_dim, 1, self.num_layers2, self.h_dim2, self.active_fn)
                for _ in range(self.t_max)
            ])
        else:
            self.heads_a0 = None

    def forward(self, x, dropout_rate=0.0):
        # Encoder
        # Manually handling dropout to match TF placeholder logic
        z = self.encoder.layers[0](x)
        if len(self.encoder.layers) > 1:
            z = self.active_fn(z)
            z = F.dropout(z, p=dropout_rate, training=self.training)
            for layer in self.encoder.layers[1:-1]:
                z = layer(z)
                z = self.active_fn(z)
                z = F.dropout(z, p=dropout_rate, training=self.training)
            z = self.encoder.layers[-1](z)

        # Batch Norm & Activation
        z = self.bn(z)
        z = self.active_fn(z)
        z = F.dropout(z, p=dropout_rate, training=self.training)

        logits_a1_list = []
        logits_a0_list = []

        # Loop through time steps
        for m in range(self.t_max):
            # A=1
            l1 = self.heads_a1[m](z) # output (B, 1)
            logits_a1_list.append(l1)

            # A=0
            if self.is_treat:
                l0 = self.heads_a0[m](z)
                logits_a0_list.append(l0)
            else:
                logits_a0_list.append(torch.zeros_like(l1))

        # Concatenate along dim 1 -> (B, T_max)
        logits_a1 = torch.cat(logits_a1_list, dim=1)
        logits_a0 = torch.cat(logits_a0_list, dim=1)

        return z, logits_a1, logits_a0

    def calculate_loss(self, x, y, t, a, w, beta, gamma, dropout_rate=0.0):
        # Forward pass
        z, logits_a1, logits_a0 = self.forward(x, dropout_rate)

        # Prepare Masks
        # t shape: (B, 1), w shape: (B, T_max, 2), a shape: (B, 1)
        tmp_range = torch.arange(0, self.t_max, device=self.device).float().unsqueeze(0) # (1, T_max)
        mask1 = (tmp_range == t).float() # (B, T_max) Equality
        mask2 = (tmp_range <= t).float() # (B, T_max) At risk

        y_expanded = mask1 * y # Broadcasting y (B, 1) -> (B, T_max) if needed, but usually y is (B, 1)
        if y.shape[1] != self.t_max:
            y_expanded = mask1 * y # Assuming y is binary event indicator

        w_clipped = torch.clamp(w, 0., self.clipping_thres)

        # --- IPM Loss ---
        loss_ipm = 0.0
        if self.ipm_term != 'no_ipm':
            for m in range(self.t_max):
                # Indices where A=1 and patient is at risk (mask2=1)
                idx1 = ((a[:, 0] * mask2[:, m]) == 1).nonzero(as_tuple=False).squeeze()

                # A=0
                idx0 = None
                if self.is_treat:
                    idx0 = (((1. - a[:, 0]) * mask2[:, m]) == 1).nonzero(as_tuple=False).squeeze()

                # IPM Term A=1
                if idx1.numel() > 0:
                    z_sub = z[idx1]
                    w_sub = w_clipped[idx1, m, 0]
                    # Compare subset z_sub with full z or control z
                    # The TF code compares z vs z[idx1]
                    if self.ipm_term == 'mmd_lin':
                        loss_ipm += mmd2_lin(z, z_sub, torch.ones_like(z[:,0]), w_sub)
                    elif self.ipm_term == "wasserstein":
                        loss_ipm += wasserstein(z, z_sub, torch.ones_like(z[:,0]), w_sub)

                # IPM Term A=0
                if self.is_treat and idx0.numel() > 0:
                    z_sub = z[idx0]
                    w_sub = w_clipped[idx0, m, 1]
                    if self.ipm_term == 'mmd_lin':
                        loss_ipm += mmd2_lin(z, z_sub, torch.ones_like(z[:,0]), w_sub)
                    elif self.ipm_term == 'wasserstein':
                        loss_ipm += wasserstein(z, z_sub, torch.ones_like(z[:,0]), w_sub)

        # --- Smoothing Loss ---
        loss_smoothing = 0.0
        if self.is_smoothing:
            for m in range(1, self.t_max):
                # Weights from current and prev step
                for p_curr, p_prev in zip(self.heads_a1[m].parameters(), self.heads_a1[m-1].parameters()):
                    loss_smoothing += torch.mean((p_curr - p_prev) ** 2)

                if self.is_treat:
                    for p_curr, p_prev in zip(self.heads_a0[m].parameters(), self.heads_a0[m-1].parameters()):
                        loss_smoothing += torch.mean((p_curr - p_prev) ** 2)

        # --- Factual Loss (Weighted Log Loss) ---
        # Normalize weights
        # w: (B, T_max, 2)
        # Denom for A=1
        denom1 = torch.sum(mask2 * a * w[:, :, 0], dim=0, keepdim=True) + 1e-8
        tmp_w1 = (w[:, :, 0] / denom1) * mask2 * a

        denom0 = 1.0
        tmp_w0 = torch.zeros_like(tmp_w1)
        if self.is_treat:
            denom0 = torch.sum(mask2 * (1. - a) * w[:, :, 1], dim=0, keepdim=True) + 1e-8
            tmp_w0 = (w[:, :, 1] / denom0) * mask2 * (1. - a)

        # Binary Cross Entropy with Logits
        # labels: y_expanded, logits: logits_A1
        bce_a1 = F.binary_cross_entropy_with_logits(logits_a1, y_expanded, reduction='none')
        loss_a1 = torch.sum(tmp_w1 * bce_a1)

        loss_factual = loss_a1
        if self.is_treat:
            bce_a0 = F.binary_cross_entropy_with_logits(logits_a0, y_expanded, reduction='none')
            loss_a0 = torch.sum(tmp_w0 * bce_a0)
            loss_factual += loss_a0

        # --- L2 Regularization (Encoder only) ---
        loss_reg = 0.0
        if self.reg_scale > 0:
            for p in self.encoder.parameters():
                loss_reg += torch.sum(p ** 2)
            loss_reg *= self.reg_scale

        total_loss = loss_factual + beta * loss_ipm + gamma * loss_smoothing + loss_reg

        return total_loss, loss_factual, loss_ipm

    # --- Training Step ---
    def train_step(self, optimizer, x, y, t, a, w, beta=1e-3, gamma=1e-3, dropout_rate=0.0):
        self.train()
        optimizer.zero_grad()

        # Convert to tensor if numpy
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        t = torch.tensor(t, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).to(self.device)
        w = torch.tensor(w, dtype=torch.float32).to(self.device)

        if not self.is_smoothing:
            gamma = 0.0

        loss_total, loss_fact, loss_ipm = self.calculate_loss(x, y, t, a, w, beta, gamma, dropout_rate)

        loss_total.backward()
        optimizer.step()

        return loss_total.item(), loss_fact.item(), loss_ipm.item()

    # --- Training Baseline ---
    def train_baseline(self, optimizer, x, y, t, a, dropout_rate=0.0):
        # Calls train_step with default weights and zero beta/gamma
        w = np.ones([x.shape[0], self.t_max, 2])
        return self.train_step(optimizer, x, y, t, a, w, beta=self.beta, gamma=0.0, dropout_rate=dropout_rate)

    # --- Inference ---
    def _predict_hazard(self, x, is_treat_group=True):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            z, logits_a1, logits_a0 = self.forward(x, dropout_rate=0.0)

            logits = logits_a1 if is_treat_group else logits_a0

            odd = torch.exp(logits)
            hazard = odd / (1. + odd)
            return hazard.cpu().numpy()

    def predict_hazard_A1(self, x):
        return self._predict_hazard(x, is_treat_group=True)

    def predict_hazard_A0(self, x):
        return self._predict_hazard(x, is_treat_group=False)

    def predict_survival_A1(self, x):
        hazard = self.predict_hazard_A1(x)
        # Cumprod of survival prob: (1 - h)
        surv = np.cumprod(1. - hazard, axis=1)
        return surv

    def predict_survival_A0(self, x):
        hazard = self.predict_hazard_A0(x)
        surv = np.cumprod(1. - hazard, axis=1)
        return surv
