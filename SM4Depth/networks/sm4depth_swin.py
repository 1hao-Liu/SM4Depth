import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .pst import PyramidSceneTransformer
from .decoder_utils import HSC_Decoder


class SM4Depth(nn.Module):
    def __init__(self, version=None, pretrained=None, frozen_stages=-1, kbins=4, **kwargs):
        super().__init__()

        self.up_mode = 'bilinear'
        self.kbins = kbins

        # Swin Transformer V1
        window_size = int(version[-2:])
        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,  
            num_heads=num_heads, 
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        self.backbone = SwinTransformer(**backbone_cfg)

        pst_cfg = dict(in_channels=in_channels[-1], out_channels=in_channels[-1], num_layers=3,
                       embedding_dim=128, num_heads=4, sizes=[(1, 1), (2, 2), (4, 4)], 
                       reg_mid_channels=256, reg_out_channels=256, kbins=kbins)
        self.pst = PyramidSceneTransformer(**pst_cfg)
        self.decoder = HSC_Decoder(num_features=in_channels, skip_channels=[48, 96, 192, 384],
                                   up_mode="bilinear")

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, imgs):
        feats = self.backbone(imgs)
        
        bins, f4, w = self.pst(feats[-1])

        w = F.softmax(w, dim=1)
        w = w.unsqueeze(-1)
        bins = nn.functional.pad(bins, (1, 0), mode='constant', value=0.01)
        bins = torch.permute(bins, (1, 0, 2))
        bins = torch.cumsum(bins, dim=2)
        bins = torch.sum(bins * w, dim=1)
        centers = 0.5 * (bins[:, :-1] + bins[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        out = self.decoder(feats[0], feats[1], feats[2], f4, centers)
        return out, centers, w.squeeze(-1)
