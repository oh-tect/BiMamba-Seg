# --------------------------------------------------------
# CAT-Seg: Cost Aggregation for Open-vocabulary Semantic Segmentation
# Licensed under The MIT License [see LICENSE for details]
# Written by Seokju Cho and Heeseong Shin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .EfficientAggregator import EffAggregatorLayer
  
class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)
        
class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, 
                 text_guidance_dim=512, 
                 appearance_guidance=512, 
                 nheads=4, 
                 input_resolution=(20, 20), 
                 pooling_size=(5, 5), 
                 window_size=(10, 10), 
                 attention_type='linear', 
                 pad_len=256,
                 clip_pretrained="ViT-B/16",
                 *, use_visual_modulation, visual_modulation_mid_amp=0.05, visual_modulation_high_amp=0.10,
                 use_text_modulation, use_mamba, mamba_scan_mode="bidi_snake") -> None:
        super().__init__()
        self.use_visual_modulation = use_visual_modulation
        self.use_text_modulation = use_text_modulation
        self.use_mamba = use_mamba
        self.eff_agg = EffAggregatorLayer(
            hidden_dim=hidden_dim,
            text_guidance_dim=text_guidance_dim,
            num_heads=nheads,
            drop_path=0.,
            sr_ratio=2,
            clip_pretrained=clip_pretrained,
            use_visual_modulation=self.use_visual_modulation,
            visual_modulation_mid_amp=visual_modulation_mid_amp,
            visual_modulation_high_amp=visual_modulation_high_amp,
            use_text_modulation=self.use_text_modulation,
            use_mamba=self.use_mamba,
            mamba_scan_mode=mamba_scan_mode,
        )
    
    def forward(self, x, appearance_guidance, text_guidance, spatial_text_guidance=None, last_layer=None, return_features=False):
        """
        Arguments:
            x: B C T H W
        """
        return self.eff_agg(
            x,
            appearance_guidance,
            text_guidance,
            spatial_text_guidance,
            last_layer,
            return_features=return_features,
        )
    
class Aggregator(nn.Module):
    def __init__(self,
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4,
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
        clip_pretrained="ViT-B/16",
        *, use_visual_modulation, visual_modulation_mid_amp=0.05, visual_modulation_high_amp=0.10,
        use_text_modulation, use_mamba, mamba_scan_mode="bidi_snake") -> None:
    
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim,
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
                clip_pretrained=clip_pretrained,
                use_visual_modulation=use_visual_modulation,
                visual_modulation_mid_amp=visual_modulation_mid_amp,
                visual_modulation_high_amp=visual_modulation_high_amp,
                use_text_modulation=use_text_modulation,
                use_mamba=use_mamba,
                mamba_scan_mode=mamba_scan_mode,
            ) for _ in range(num_layers)
        ])
                
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.pad_len = pad_len

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        if x.dim() == 5:
            B = x.shape[0]
            corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        elif x.dim() == 4:
            if guidance is None or guidance[0] is None:
                raise ValueError('Cannot infer batch size from 4D input without guidance')
            B = guidance[0].shape[0]
            corr_embed = x
        else:
            raise ValueError(f'Unsupported corr_embed shape: {x.shape}')

        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

    def forward(self, img_feats, global_text_feats, text_feats, vis_guidance, last_layer=None, return_features=False):
        """
        Arguments:
            img_feats: (B, C, H, W)
            global_text_feats: (B, Cg) clean text features, directly projected as spatial_text_guidance
            text_feats: (B, T, P, C) raw text features, projected as text_guidance
            vis_guidance: list of appearance guidance tensors [[B, C, H, W], ...]
            last_layer: (B, C, H, W) last CLIP layer feature passed separately
        """
        corr = self.correlation(img_feats, text_feats) # B P T H W

        projected_guidance, projected_text_guidance, projected_spatial_text_guidance, projected_decoder_guidance = None, None, None, [None, None]
        if self.guidance_projection is not None and vis_guidance is not None and len(vis_guidance) > 0:
            projected_guidance = self.guidance_projection(vis_guidance[0])
        if self.decoder_guidance_projection is not None and vis_guidance is not None and len(vis_guidance) > 1:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, vis_guidance[1:])]

        # Project text features to the dimension required by the decoder (if projection is configured)
        projected_text_guidance = None
        if self.text_guidance_projection is not None:
            # text_feats: [B, T, P, C]
            t = text_feats.mean(dim=-2)  # (B, T, C)
            projected_text_guidance = self.text_guidance_projection(t)
            projected_spatial_text_guidance = global_text_feats
        else:
            # If no projection, average over the prompt dimension, keep (B, T, C)
            projected_text_guidance = text_feats.mean(dim=-2)
            projected_spatial_text_guidance = global_text_feats
            
        corr_embed = self.corr_embed(corr)
        spatial_features = None
        for layer in self.layers:
            if return_features:
                corr_embed, spatial_features = layer(
                    corr_embed,
                    projected_guidance,
                    projected_text_guidance,
                    projected_spatial_text_guidance,
                    last_layer,
                    return_features=True,
                )
            else:
                corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance, projected_spatial_text_guidance, last_layer)
        
        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        if return_features:
            return logit, spatial_features
        return logit
