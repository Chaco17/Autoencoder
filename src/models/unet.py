"""U-Net autoencoder: encoder-decoder with skip connections between equal resolutions.

The bottleneck is an 8x8 map with latent_dim channels; its global average is used as z (for t-SNE).
"""
import torch
import torch.nn as nn

NUM_LEVELS = 4  # 128 -> 64 -> 32 -> 16 -> 8 (bottleneck)


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels=3, img_size=128, latent_dim=128, base_channels=32):
        super().__init__()
        assert img_size % 2**NUM_LEVELS == 0, f"img_size must be divisible by {2**NUM_LEVELS}"
        channels = [base_channels * 2**i for i in range(NUM_LEVELS)]  # 32, 64, 128, 256

        self.inc = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(c_in, c_out)) for c_in, c_out in zip(channels, channels[1:])
        )
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(channels[-1], latent_dim))

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        in_ch = latent_dim
        for out_ch in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(out_ch * 2, out_ch))  # *2: concatenated skip connection
            in_ch = out_ch

        self.out = nn.Sequential(nn.Conv2d(channels[0], in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        h = self.bottleneck(skips[-1])
        z = h.mean(dim=(2, 3))

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            h = conv(torch.cat([up(h), skip], dim=1))
        return {"recon": self.out(h), "z": z, "mu": None, "logvar": None}
