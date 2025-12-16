import torch
import torch.nn.functional as F
import numpy as np

_EPSILON = 1e-08


################################
##### USER-DEFINED FUNCTIONS
def log(x):
    return torch.log(x + _EPSILON)

def div(x, y):
    return x / (y + _EPSILON)

################################
##### IPM TERMS
def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    # X: (n1, d), Y: (n2, d)
    C = -2 * torch.matmul(X, Y.t())
    nx = torch.sum(X**2, dim=1, keepdim=True)
    ny = torch.sum(Y**2, dim=1, keepdim=True)
    D = (C + ny.t()) + nx
    return D


def mmd2_lin(X1, X2, W1=None, W2=None, p=0.5, weights=None):
    ''' Linear MMD '''    
    if (W1 is None) and (W2 is None):
        W1 = torch.ones_like(X1[:, 0])
        W2 = torch.ones_like(X2[:, 0])
    
    W1 = div(W1, torch.sum(W1))
    W2 = div(W2, torch.sum(W2))
    
    W1 = W1.view(-1, 1)
    W2 = W2.view(-1, 1)
        
    mean1 = torch.sum(W1 * X1, dim=0)
    mean2 = torch.sum(W2 * X2, dim=0)
    
    mmd = torch.sum((2.0 * p * mean1 - 2.0 * (1.0 - p) * mean2)**2)
    
    return mmd


def wasserstein(X1, X2, W1=None, W2=None, p=0.5, lam=10, its=10):
    """ Returns the Wasserstein distance between treatment groups """    
    device = X1.device
    dtype = X1.dtype
    
    n1 = float(X1.shape[0])
    n2 = float(X2.shape[0])
    
    ''' Compute distance matrix'''
    M = pdist2sq(X1, X2)
        
    # for now consider W1 and W2 is [None,] shape
    if (W1 is None) and (W2 is None):
        W1 = torch.ones_like(X1[:, 0])
        W2 = torch.ones_like(X2[:, 0])
    
    W1 = div(W1, torch.sum(W1))
    W2 = div(W2, torch.sum(W2))
    
    W1 = W1.view(-1, 1)
    W2 = W2.view(-1, 1)
    
    # Outer product for mask
    W_mask = W1.repeat(1, int(n2)) * W2.t().repeat(int(n1), 1)
    
    ''' Estimate lambda and delta '''
    M_mean = torch.sum(M * W_mask) # this becomes weighted average
    
    # Note: tf.nn.dropout rate is probability to drop. 
    # PyTorch dropout p is also probability to zero out.
    # The logic provided in TF was rate=1-(...), so p=1-(...).
    # M_drop is calculated but not used in the original TF calculation for D. 
    # Kept here for strict translation consistency.
    drop_prob = 1.0 - (10.0 / (n1 * n2))
    if drop_prob < 0: drop_prob = 0.0
    M_drop = F.dropout(M, p=drop_prob, training=True) 
    
    delta = torch.max(M).detach()
    eff_lam = (lam / M_mean).detach()

    ''' Compute new distance matrix '''
    # Expansion: Add row and col of deltas, with 0 at the corner
    # Row extension
    row_ext = delta * torch.ones((1, M.shape[1]), device=device, dtype=dtype)
    Mt = torch.cat([M, row_ext], dim=0)
    
    # Col extension (including the corner zero)
    col_ext = delta * torch.ones((M.shape[0], 1), device=device, dtype=dtype)
    corner = torch.zeros((1, 1), device=device, dtype=dtype)
    col = torch.cat([col_ext, corner], dim=0)
    
    Mt = torch.cat([Mt, col], dim=1)

    ''' Compute marginal vectors '''        
    # a: concat [p * W1, (1-p)]
    a_top = p * torch.ones_like(X1[:, 0:1]) * W1
    a_bot = (1 - p) * torch.ones((1, 1), device=device, dtype=dtype)
    a = torch.cat([a_top, a_bot], dim=0)

    # b: concat [(1-p) * W2, p]
    b_top = (1 - p) * torch.ones_like(X2[:, 0:1]) * W2
    b_bot = p * torch.ones((1, 1), device=device, dtype=dtype)
    b = torch.cat([b_top, b_bot], dim=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6 # added constant to avoid nan
    # U = K * Mt # Not used in subsequent calculations in original code
    ainvK = K / a

    u = a
    for i in range(0, its):
        # TF: u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
        # Decomposing inner matmuls for shapes (assuming u is col vector)
        # u.t() @ K -> (1, m)
        denom = b / (torch.matmul(u.t(), K).t())
        u = 1.0 / torch.matmul(ainvK, denom)
        
    v = b / (torch.matmul(u.t(), K).t())

    T = u * (v.t() * K)

    E = T * Mt
    D = 2 * torch.sum(E)

    return D
