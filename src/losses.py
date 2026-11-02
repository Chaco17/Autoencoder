"""Reconstruction losses not available in torch.nn (L1 and L2 use nn.L1Loss / nn.MSELoss).

Images are expected in [0, 1], hence data_range=1.0.
"""
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


class SSIMLoss(nn.Module):
    """1 - SSIM: 0 when both images are structurally identical."""

    def forward(self, recon, target):
        return 1.0 - ssim(recon, target, data_range=1.0)


class SSIML1Loss(nn.Module):
    """alpha * (1 - SSIM) + (1 - alpha) * L1: structure from SSIM, colour/intensity from L1."""

    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, recon, target):
        ssim_term = 1.0 - ssim(recon, target, data_range=1.0)
        return self.alpha * ssim_term + (1.0 - self.alpha) * F.l1_loss(recon, target)
