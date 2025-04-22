import einops
import lietorch
import torch
import numpy as np


def as_SE3(X):
    if isinstance(X, lietorch.SE3):
        return X
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC

def as_Sim3(X, s=None, device='cuda'):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(device)
    
    if s != None:
        scale = s.data[:,-1:]
        X = X.unsqueeze(0)
        X = torch.cat([X, scale], dim=1)
        X.requires_grad_(True)
        T_WC = lietorch.Sim3(X)
    else:
        T_WC = lietorch.Sim3(lietorch.SE3(X.unsqueeze(0)))
    return T_WC