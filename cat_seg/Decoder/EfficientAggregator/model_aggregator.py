import torch
import math
import random
import torch.nn as nn

from einops import rearrange, repeat
from .model_agg_class import ClassFusionBlock

  
class EffAggregatorLayer(nn.Module):
    def __init__(self, hidden_dim, text_guidance_dim=None, num_heads=8, drop_path=0., sr_ratio=2, clip_pretrained="ViT-B/16", *, use_visual_modulation, visual_modulation_mid_amp=0.05, visual_modulation_high_amp=0.10, use_text_modulation, use_mamba, mamba_scan_mode="bidi_snake"):
        super().__init__()
        self.use_mamba = use_mamba
        if self.use_mamba:
            from .model_agg_spatial import SptialFusionBlock as MambaSptialFusionBlock

            self.spatial_fusion = MambaSptialFusionBlock(
                hidden_dim,
                num_heads,
                drop_path=drop_path,
                sr_ratio=sr_ratio,
                text_dim=text_guidance_dim,
                use_visual_modulation=use_visual_modulation,
                visual_modulation_mid_amp=visual_modulation_mid_amp,
                visual_modulation_high_amp=visual_modulation_high_amp,
                scan_mode=mamba_scan_mode,
            )
        else:
            from .model_agg_spatial_trans import SptialFusionBlock as TransSptialFusionBlock

            self.spatial_fusion = TransSptialFusionBlock(
                hidden_dim,
                num_heads,
                drop_path=drop_path,
                sr_ratio=sr_ratio,
                text_dim=text_guidance_dim,
                use_visual_modulation=use_visual_modulation,
            )
        self.class_fusion = ClassFusionBlock(hidden_dim, num_heads, drop_path=drop_path, clip_pretrained=clip_pretrained, use_text_modulation=use_text_modulation)

    def forward(self, x, img_guidance, text_guidance, spatial_text_guidance=None, last_layer=None, return_features=False):
        # Support both shapes: (B, C, T, H, W) or ((B*T), C, H, W)
        if x.dim() == 4:
            # x is likely (B*T) C H W - infer B from guidance if possible
            if img_guidance is not None:
                B = img_guidance.shape[0]
            elif spatial_text_guidance is not None:
                B = spatial_text_guidance.shape[0]
            elif text_guidance is not None:
                B = text_guidance.shape[0]
            else:
                raise ValueError("Cannot infer batch size from inputs when x is 4D")
            T = x.shape[0] // B
            x = rearrange(x, '(B T) C H W -> B C T H W', B=B, T=T)

        B, _, T, H, W = x.shape

        # spatial fusion: reshape to (B*T, C, H, W)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        if img_guidance is not None:
            img_guidance = repeat(img_guidance, "B C H W -> (B T) C H W", T=T)
        text_guidance_spatial = spatial_text_guidance if spatial_text_guidance is not None else text_guidance
        if text_guidance_spatial is not None:
            if text_guidance_spatial.dim() == 3:
                if text_guidance_spatial.shape[0] != B or text_guidance_spatial.shape[1] != T:
                    raise ValueError(
                        f'Expected spatial text guidance shape [B, T, C] with B={B}, T={T}, '
                        f'but got {text_guidance_spatial.shape}'
                    )
                class_idx = torch.arange(T, device=text_guidance_spatial.device).repeat(B)
                text_guidance_spatial = {
                    'own': rearrange(text_guidance_spatial, 'B T C -> (B T) 1 C'),
                    'all': repeat(text_guidance_spatial, 'B N C -> (B T) N C', T=T),
                    'class_idx': class_idx,
                }
            elif text_guidance_spatial.dim() == 2:
                if text_guidance_spatial.shape[0] != B:
                    raise ValueError(
                        f'Expected spatial text guidance shape [B, C] with B={B}, '
                        f'but got {text_guidance_spatial.shape}'
                    )
                text_guidance_spatial = repeat(text_guidance_spatial, 'B C -> (B T) 1 C', T=T)
            else:
                raise ValueError(f'Unsupported text guidance shape: {text_guidance_spatial.shape}')
        if return_features and self.use_mamba:
            x, spatial_features = self.spatial_fusion(
                x,
                img_guidance,
                text_guidance_spatial,
                return_features=True,
            )
        elif return_features:
            x, spatial_features = self.spatial_fusion(
                x,
                img_guidance,
                text_guidance_spatial,
                return_features=True,
            )
        elif self.use_mamba:
            x = self.spatial_fusion(x, img_guidance, text_guidance_spatial)
        else:
            x = self.spatial_fusion(x, img_guidance, text_guidance_spatial)

        # class fusion
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        x = self.class_fusion(x, text_guidance, last_layer)

        if return_features:
            return x, spatial_features
        return x
