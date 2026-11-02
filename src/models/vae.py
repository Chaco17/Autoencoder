"""Convolutional variational autoencoder.

128x128 -> 4 stride-2 convs -> 8x8 feature map -> (mu, logvar) of size latent_dim -> z -> decoder.
"""
import torch
import torch.nn as nn

NUM_BLOCKS = 4  # each block halves the resolution: 128 -> 64 -> 32 -> 16 -> 8


def down_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class VAE(nn.Module):
    def __init__(self, in_channels=3, img_size=128, latent_dim=128, base_channels=32):
        super().__init__()
        assert img_size % 2**NUM_BLOCKS == 0, f"img_size must be divisible by {2**NUM_BLOCKS}"
        channels = [base_channels * 2**i for i in range(NUM_BLOCKS)]  # 32, 64, 128, 256
        self.feat_ch = channels[-1]
        self.feat_size = img_size // 2**NUM_BLOCKS
        flat_dim = self.feat_ch * self.feat_size**2

        self.encoder = nn.Sequential(*[down_block(c_in, c_out) for c_in, c_out in zip([in_channels] + channels, channels)])
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        rev = channels[::-1]  # 256, 128, 64, 32
        self.decoder = nn.Sequential(
            *[up_block(c_in, c_out) for c_in, c_out in zip(rev, rev[1:] + [rev[-1]])],
            nn.Conv2d(rev[-1], in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # outputs in [0, 1], same range as the input images
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def reparameterize(self, mu, logvar):
        """z = mu + sigma * eps, so sampling stays differentiable."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.feat_ch, self.feat_size, self.feat_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return {"recon": self.decode(z), "z": mu, "mu": mu, "logvar": logvar}
