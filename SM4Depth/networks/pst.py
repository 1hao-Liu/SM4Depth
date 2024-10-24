import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class adaptive_pst_kbins(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, patch_size=(), num_heads=4, num_layers=4, kbins=6,
                 h=14, w=18):  # h, w is related to the input resolution, which is a bug waiting to be resolved 
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers) 
        self.kbins = kbins

        if patch_size == (2, 2) or patch_size == (4, 4):
            self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim, patch_size, patch_size, (h % 2, w % 2))
        else:
            self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim, patch_size, patch_size, 0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2) 
        embeddings = nn.functional.pad(embeddings, (self.kbins + 1, 0, 0, 0, 0, 0))
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)

        return x


class PyramidSceneTransformer(nn.Module):
    def __init__(self, in_channels=768, out_channels=256, num_layers=3, embedding_dim=128, 
                 num_heads=4, sizes=[(1, 1), (2, 2), (4, 4)], reg_mid_channels=256, 
                 reg_out_channels=256, kbins=6, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers  
        self.embedding_dim = embedding_dim 
        self.num_heads = num_heads  
        self.sizes = sizes
        self.kbins = kbins
        self.stages = nn.ModuleList(
            [self._make_stage(self.in_channels, self.embedding_dim, size, self.num_heads, self.num_layers, self.kbins) for size in
             self.sizes]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.embedding_dim * len(self.sizes), self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, reg_mid_channels),
            nn.LeakyReLU(),
            nn.Linear(reg_mid_channels, reg_mid_channels),  
            nn.LeakyReLU(), 
            nn.Linear(reg_mid_channels, reg_out_channels), 
        )

        self.head = nn.Linear(self.embedding_dim, self.kbins)

    def _make_stage(self, in_channels, embedding_dim, size, num_heads, num_layers, kbins):
        transformer = adaptive_pst_kbins(in_channels, embedding_dim, size, num_heads, num_layers, kbins)

        return nn.Sequential(transformer)

    def forward(self, x):
        b, c, h, w = x.shape
        priors = [stage(x) for stage in self.stages]
        bins = [priors[0][i, ...].unsqueeze(0) for i in range(self.kbins)]
        cls_weight = priors[0][self.kbins, ...]

        priors = [prior[self.kbins + 1:, ...] for prior in priors]

        for i in range(len(priors)):
            priors[i] = priors[i].permute(1, 2, 0).reshape(b, self.embedding_dim, math.ceil(h / self.sizes[i][0]), -1)
            priors[i] = F.interpolate(priors[i], size=(h, w), mode='nearest')

        priors = self.bottleneck(torch.cat(priors, 1))
        bins = torch.cat(bins)
        bins = self.regressor(bins) + 0.1
        w = self.head(cls_weight)

        return bins, priors, w
