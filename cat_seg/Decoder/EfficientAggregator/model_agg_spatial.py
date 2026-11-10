import torch
import math

from torch import nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        x = rearrange(x, 'B H W C -> B C H W')
        return x


class RowRecurrentScanner(nn.Module):
    def __init__(self, d_model, num_heads=4, gamma_range=(0.6, 0.98),
                 fuse_mode='gate', dropout=0.0, bidi=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d = d_model
        self.h = num_heads
        self.dh = d_model // num_heads
        self.bidi = bidi
        self.fuse_mode = fuse_mode

        self.out_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.proj_in = nn.Linear(d_model, 3 * d_model)
        self.alpha_logit = nn.Parameter(torch.zeros(d_model))
        self.mix_w = nn.Parameter(torch.zeros(d_model))
        self.drop = nn.Dropout(dropout)

        lo, hi = gamma_range
        gammas = torch.linspace(lo, hi, steps=self.h)
        self.register_buffer("gamma_heads", gammas, persistent=False)

        if self.bidi and self.fuse_mode == 'gate':
            self.fuse_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )

    def _decay_coeff(self):
        sig = torch.sigmoid
        alpha_l = sig(self.alpha_logit)
        w = sig(self.mix_w)
        g = self.gamma_heads.repeat_interleave(self.dh)
        decay_eff = w * alpha_l + (1 - w) * g
        return decay_eff.view(1, 1, self.d)

    def _recur_1d(self, x, decay_eff, direction='fwd', init_state=None):
        B, L, D = x.shape
        x = self.out_norm(x)
        gate = torch.sigmoid(self.gate(x))
        beta, gamma, delta = self.proj_in(x).chunk(3, dim=-1)
        beta = torch.sigmoid(beta)

        s = x.new_zeros(B, D) if init_state is None else init_state
        if direction == 'rev':
            x = torch.flip(x, [1])
            gate, beta, gamma, delta = map(lambda t: torch.flip(t, [1]),
                                           (gate, beta, gamma, delta))

        a = decay_eff.squeeze(0).squeeze(0)
        ys = []
        for t in range(L):
            s = a * s + beta[:, t] * x[:, t]
            y_t = torch.tanh(gamma[:, t]) * s + delta[:, t]
            y_t = gate[:, t] * y_t + (1 - gate[:, t]) * x[:, t]
            ys.append(y_t)

        y = torch.stack(ys, 1)
        if direction == 'rev':
            y = torch.flip(y, [1])
        return y, s

    def forward(self, x, init_state_fwd=None, init_state_rev=None):
        decay_eff = self._decay_coeff()
        y_f, s_f = self._recur_1d(x, decay_eff, 'fwd', init_state_fwd)
        if not self.bidi:
            return self.drop(y_f), s_f, None

        y_b, s_b = self._recur_1d(x, decay_eff, 'rev', init_state_rev)
        if self.fuse_mode == 'avg':
            y = 0.5 * (y_f + y_b)
        else:
            g = torch.sigmoid(self.fuse_gate(x))
            y = g * y_f + (1 - g) * y_b
        return self.drop(y), s_f, s_b

import torch.nn.functional as F
    
class VisualModulator(nn.Module):

    def __init__(
        self,
        text_dim,
        vis_dim,
        init_scale=0.01,
        mid_amp=0.05,
        high_amp=0.10,
    ):
        super().__init__()
        self.vis_dim = vis_dim
        self.text_dim = text_dim

        self.low_cut_param = nn.Parameter(torch.tensor(0.0))
        self.high_cut_param = nn.Parameter(torch.tensor(0.0))
        self.band_sharpness = 40.0

        self.mid_amp = float(mid_amp)
        self.high_amp = float(high_amp)

        # Original global text modulation head
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, vis_dim),
            nn.GELU(),
            nn.Linear(vis_dim, vis_dim * 2)  # gamma_mid, gamma_high
        )

        self.class_freq_pref = nn.Linear(text_dim, 1)
        if text_dim == vis_dim:
            self.text_mask_proj = nn.Identity()
        else:
            self.text_mask_proj = nn.Linear(text_dim, vis_dim, bias=False)

        # Initialize as identity mapping
        nn.init.zeros_(self.text_proj[-1].weight)
        nn.init.zeros_(self.text_proj[-1].bias)

        # Initial class frequency preference is 0, so mid/high scale maps are both 1
        nn.init.zeros_(self.class_freq_pref.weight)
        nn.init.zeros_(self.class_freq_pref.bias)

        self.mod_scale = nn.Parameter(torch.tensor(float(init_scale)))

        # Fixed temperature, not exposed as hyperparameter
        self.mask_tau = 0.10

        # debug
        self.latest_pref = None
        self.latest_mid_scale = None
        self.latest_high_scale = None
        self.latest_low_cut = None
        self.latest_high_cut = None
        self.latest_mask_entropy = None
        self.latest_mask_conf = None

    def _prepare_text_guidance(self, text_guidance, B):
        """
        Convert text guidance to [B, N, D].

        Supported:
            [N, D]
            [B, N, D]
            [B, N, P, D]
        """
        if text_guidance.dim() == 2:
            # fused_per_class: [N, D] -> [B, N, D]
            text_guidance = text_guidance.unsqueeze(0).expand(B, -1, -1)

        elif text_guidance.dim() == 3:
            # [B, N, D]
            if text_guidance.shape[0] != B:
                if text_guidance.shape[0] == 1:
                    text_guidance = text_guidance.expand(B, -1, -1)
                else:
                    raise ValueError(
                        f"Batch mismatch: text_guidance.shape={text_guidance.shape}, B={B}"
                    )

        elif text_guidance.dim() == 4:
            # [B, N, P, D] -> average prompts only, keep class dimension
            if text_guidance.shape[0] != B:
                if text_guidance.shape[0] == 1:
                    text_guidance = text_guidance.expand(B, -1, -1, -1)
                else:
                    raise ValueError(
                        f"Batch mismatch: text_guidance.shape={text_guidance.shape}, B={B}"
                    )
            text_guidance = text_guidance.mean(dim=2)

        else:
            raise ValueError(
                f"Unsupported text guidance shape: {text_guidance.shape}"
            )

        return text_guidance

    def _get_modulation_text(self, text_guidance):
        if isinstance(text_guidance, dict):
            return text_guidance["own"]
        return text_guidance

    def _get_mask_text_and_class_idx(self, text_guidance):
        if isinstance(text_guidance, dict):
            return text_guidance["all"], text_guidance.get("class_idx", None)
        return text_guidance, None

    def _get_text_anchor(self, text_guidance, B):
        if text_guidance.dim() == 2:
            # [N, D] -> mean -> [B, D]
            return text_guidance.mean(dim=0, keepdim=True).expand(B, -1)

        elif text_guidance.dim() == 3:
            # [B, N, D] -> [B, D]
            if text_guidance.shape[0] != B:
                if text_guidance.shape[0] == 1:
                    text_guidance = text_guidance.expand(B, -1, -1)
                else:
                    raise ValueError(
                        f"Batch mismatch: text_guidance.shape={text_guidance.shape}, B={B}"
                    )
            return text_guidance.mean(dim=1)

        elif text_guidance.dim() == 4:
            # [B, N, P, D] -> [B, D]
            if text_guidance.shape[0] != B:
                if text_guidance.shape[0] == 1:
                    text_guidance = text_guidance.expand(B, -1, -1, -1)
                else:
                    raise ValueError(
                        f"Batch mismatch: text_guidance.shape={text_guidance.shape}, B={B}"
                    )
            return text_guidance.mean(dim=(1, 2))

        else:
            raise ValueError(
                f"Unsupported text guidance shape: {text_guidance.shape}"
            )

    def _make_freq_masks(self, H, W, device, dtype):
        fy = torch.fft.fftfreq(H, device=device).view(H, 1)
        fx = torch.fft.fftfreq(W, device=device).view(1, W)
        radius = torch.sqrt(fx ** 2 + fy ** 2)

        low_cut = 0.10 + 0.10 * torch.sigmoid(self.low_cut_param)
        high_cut = 0.30 + 0.10 * torch.sigmoid(self.high_cut_param)

        k = self.band_sharpness

        mid_mask = torch.sigmoid(k * (radius - low_cut)) * torch.sigmoid(k * (high_cut - radius))
        high_mask = torch.sigmoid(k * (radius - high_cut))

        with torch.no_grad():
            self.latest_low_cut = low_cut.detach().cpu()
            self.latest_high_cut = high_cut.detach().cpu()

        return (
            mid_mask.to(dtype).view(1, 1, H, W),
            high_mask.to(dtype).view(1, 1, H, W),
        )

    def _semantic_freq_scale_maps(self, x_float, text_guidance):
        """
        x_float: [B, N, H, W]  (N = class / cost dimension)

        We remove redundant pixel-text similarity and use cost confidence
        (top1 - top2 margin) as the spatial prior for modulation.
        """
        B, N, H, W = x_float.shape
        text_cls = self._prepare_text_guidance(
            self._get_modulation_text(text_guidance), B
        )                                                    # [B, N, D]
        class_pref = torch.tanh(self.class_freq_pref(text_cls)).view(B, -1, 1, 1)
        prob = F.softmax(x_float / self.mask_tau, dim=1)     # [B, N, H, W]
        if N >= 2:
            top2 = torch.topk(prob, k=2, dim=1).values
            spatial_gate = top2[:, 0:1] - top2[:, 1:2]       # [B, 1, H, W], margin
        else:
            spatial_gate = prob.max(dim=1, keepdim=True).values
        if getattr(self, "gate_detach", False):
            spatial_gate = spatial_gate.detach()
        pref_map = (prob * class_pref).sum(dim=1, keepdim=True) * spatial_gate
        mid_scale_map = 1.0 - self.mid_amp * pref_map
        high_scale_map = 1.0 + self.high_amp * pref_map

        with torch.no_grad():
            self.latest_pref = pref_map.mean().cpu()
            self.latest_mask_conf = spatial_gate.mean().cpu()

        return mid_scale_map, high_scale_map

    
    def forward(self, x, text_guidance):
        """
        x: [B, C, H, W]

        text_guidance:
            [N, D]        fused_per_class
            [B, N, D]
            [B, N, P, D]
            dict(own=[B, 1, D], all=[B, N, D], class_idx=[B])
        """
        B, C, H, W = x.shape

        if C != self.vis_dim:
            raise ValueError(
                f"Visual channel mismatch: x has C={C}, but vis_dim={self.vis_dim}"
            )

        t = self._get_text_anchor(self._get_modulation_text(text_guidance), B)

        gamma_mid, gamma_high = self.text_proj(t).chunk(2, dim=-1)

        gamma_mid = torch.tanh(gamma_mid).view(B, C, 1, 1)
        gamma_high = torch.tanh(gamma_high).view(B, C, 1, 1)

        mid_mask, high_mask = self._make_freq_masks(H, W, x.device, x.dtype)

        x_float = x.float()
        freq = torch.fft.fft2(x_float, dim=(-2, -1), norm="ortho")

        delta_mid_f = freq * gamma_mid.float() * mid_mask.float()
        delta_high_f = freq * gamma_high.float() * high_mask.float()

        delta_mid = torch.fft.ifft2(delta_mid_f, dim=(-2, -1), norm="ortho").real
        delta_high = torch.fft.ifft2(delta_high_f, dim=(-2, -1), norm="ortho").real

        mid_scale_map, high_scale_map = self._semantic_freq_scale_maps(
            x_float,
            text_guidance,
        )

        delta = mid_scale_map * delta_mid + high_scale_map * delta_high
        delta = delta.to(dtype=x.dtype)

        return x + self.mod_scale.to(dtype=x.dtype) * delta

class VisionMambaMixer(nn.Module):
    def __init__(self, dim, text_dim=None, *, use_visual_modulation, visual_modulation_mid_amp=0.05, visual_modulation_high_amp=0.10, drop=0.0, chunk_len=12, cross_row_gain=1.0, cross_col_gain=1.0, bidi=False, scan_mode="bidi_snake"):
        super().__init__()
        self.dim = dim
        self.chunk_len = chunk_len
        self.use_visual_modulation = use_visual_modulation
        self.film_modulator = VisualModulator(
            text_dim=text_dim or dim,
            vis_dim=dim,
            init_scale=0.01,
            mid_amp=visual_modulation_mid_amp,
            high_amp=visual_modulation_high_amp,
        ) if use_visual_modulation else None
        self.guidance_mod_logit = nn.Parameter(torch.tensor(-1.8))
        self.guidance_gate = nn.Conv2d(2 * dim, 1, kernel_size=1)
        nn.init.zeros_(self.guidance_gate.weight)
        nn.init.constant_(self.guidance_gate.bias, -2.0)
        self.pre_norm = LayerNorm2d(2 * dim)
        self.pre_proj = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=1, padding=0)
        self.pre_dwconv = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1,
                                    groups=2 * dim, bias=True)
        self.in_proj = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=1, padding=0)
        self.seq_row = RowRecurrentScanner(
            d_model=dim,
            num_heads=4,
            gamma_range=(0.6, 0.98),
            fuse_mode='gate',
            dropout=drop,
            bidi=bidi,
        )

        self.seq_col = RowRecurrentScanner(
            d_model=dim,
            num_heads=4,
            gamma_range=(0.6, 0.98),
            fuse_mode='gate',
            dropout=drop,
            bidi=bidi,
        )

        self.cross_row_gain = nn.Parameter(torch.tensor(float(cross_row_gain)))
        self.cross_col_gain = nn.Parameter(torch.tensor(float(cross_col_gain)))

        self.out_norm = nn.BatchNorm2d(dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)
        if self.film_modulator is not None:
            nn.init.zeros_(self.film_modulator.text_proj[-1].weight)
            nn.init.zeros_(self.film_modulator.text_proj[-1].bias)
            nn.init.zeros_(self.film_modulator.class_freq_pref.weight)
            nn.init.zeros_(self.film_modulator.class_freq_pref.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    @torch.no_grad()
    def _make_cut_points(self, width: int):
        idx = list(range(0, width, self.chunk_len)) if self.chunk_len > 0 else [0]
        if idx[-1] != width:
            idx.append(width)
        return idx

    def _scan_rows_chunked(self, x2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x2d.shape
        cuts = self._make_cut_points(W)
        out = x2d.new_empty(B, H, W, C)

        state_fwd = x2d.new_zeros(B, H, C)
        state_rev = x2d.new_zeros(B, H, C)

        gain = self.cross_row_gain if isinstance(self.cross_row_gain, torch.Tensor) \
            else x2d.new_tensor(self.cross_row_gain)

        odd_rows = torch.arange(1, H, 2, device=x2d.device) if H > 1 else None

        for s, e in zip(cuts[:-1], cuts[1:]):
            L = e - s
            # [B,C,H,L] -> [B,H,L,C]
            seq_bhlc = x2d[:, :, :, s:e].permute(0, 2, 3, 1).contiguous()
            seq_snake = seq_bhlc.clone()
            if odd_rows is not None and odd_rows.numel() > 0:
                seq_snake[:, odd_rows] = torch.flip(seq_bhlc[:, odd_rows], dims=[2])

            seq = seq_snake.view(B * H, L, C)
            init_f = state_fwd.view(B * H, C)
            init_r = state_rev.view(B * H, C)

            y_flat, s_f_flat, s_r_flat = self.seq_row(
                seq, init_state_fwd=init_f, init_state_rev=init_r
            )

            y_bhlc = y_flat.view(B, H, L, C)
            y_snake = y_bhlc.clone()
            if odd_rows is not None and odd_rows.numel() > 0:
                y_snake[:, odd_rows] = torch.flip(y_bhlc[:, odd_rows], dims=[2])

            out[:, :, s:e, :] = y_snake
            s_f = s_f_flat.view(B, H, C)
            s_r = s_r_flat.view(B, H, C) if s_r_flat is not None else None
            s_f_down = torch.zeros_like(s_f)
            s_f_down[:, 1:, :] = s_f[:, :-1, :]
            state_fwd = s_f + gain * s_f_down

            if s_r is not None:
                s_r_down = torch.zeros_like(s_r)
                s_r_down[:, 1:, :] = s_r[:, :-1, :]
                state_rev = s_r + gain * s_r_down
            else:
                state_rev = state_rev

        return out.permute(0, 3, 1, 2).contiguous()

    def _scan_cols_chunked(self, x2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x2d.shape
        cuts = self._make_cut_points(H)
        out = x2d.new_empty(B, H, W, C)

        state_fwd = x2d.new_zeros(B, W, C)
        state_rev = x2d.new_zeros(B, W, C)

        gain = self.cross_col_gain if isinstance(self.cross_col_gain, torch.Tensor) \
            else x2d.new_tensor(self.cross_col_gain)

        odd_cols = torch.arange(1, W, 2, device=x2d.device) if W > 1 else None

        for s, e in zip(cuts[:-1], cuts[1:]):
            L = e - s
            # [B,C,L,W] -> [B,W,L,C]
            seq_bwlc = x2d[:, :, s:e, :].permute(0, 3, 2, 1).contiguous()

            seq_snake = seq_bwlc.clone()
            if odd_cols is not None and odd_cols.numel() > 0:
                seq_snake[:, odd_cols] = torch.flip(seq_bwlc[:, odd_cols], dims=[2])

            seq = seq_snake.view(B * W, L, C)
            init_f = state_fwd.view(B * W, C)
            init_r = state_rev.view(B * W, C)

            y_flat, s_f_flat, s_r_flat = self.seq_col(
                seq, init_state_fwd=init_f, init_state_rev=init_r
            )

            y_bwlc = y_flat.view(B, W, L, C)
            y_snake = y_bwlc.clone()
            if odd_cols is not None and odd_cols.numel() > 0:
                y_snake[:, odd_cols] = torch.flip(y_bwlc[:, odd_cols], dims=[2])

            out[:, s:e, :, :] = y_snake.permute(0, 2, 1, 3).contiguous()

            s_f = s_f_flat.view(B, W, C)
            s_r = s_r_flat.view(B, W, C) if s_r_flat is not None else None
            s_f_right = torch.zeros_like(s_f)
            s_f_right[:, 1:, :] = s_f[:, :-1, :]
            state_fwd = s_f + gain * s_f_right

            if s_r is not None:
                s_r_right = torch.zeros_like(s_r)
                s_r_right[:, 1:, :] = s_r[:, :-1, :]
                state_rev = s_r + gain * s_r_right
            else:
                state_rev = state_rev

        return out.permute(0, 3, 1, 2).contiguous()

    def _scan_2d(self, x):
        y_row = self._scan_rows_chunked(x)
        y_col = self._scan_cols_chunked(x)
        return 0.5 * (y_row + y_col)

    def forward(self, x, guidance, text_guidance, return_features=False):
        x_before_modulation = x
        if self.use_visual_modulation and text_guidance is not None and self.film_modulator is not None:
            x_modulated = self.film_modulator(x_before_modulation, text_guidance)
            delta = x_modulated - x_before_modulation
            eta = torch.sigmoid(self.guidance_mod_logit).to(dtype=guidance.dtype)
            gate_input = torch.cat([guidance.detach(), delta.detach()], dim=1)
            gate = torch.sigmoid(self.guidance_gate(gate_input)).to(dtype=guidance.dtype)
            guidance_modulated = guidance + eta * gate * delta
        else:
            x_modulated = x_before_modulation
            guidance_modulated = guidance

        xg = torch.cat([x_before_modulation, guidance_modulated], dim=1)
        xg = self.pre_norm(xg)
        xg = self.pre_proj(xg)
        xg = self.pre_dwconv(xg)
        xg = self.in_proj(xg)
        u, gate = torch.chunk(xg, chunks=2, dim=1)
        gate = torch.sigmoid(gate)
        u = u * gate
        residual = u
        y = self._scan_2d(u)
        y = self.out_norm(y)
        y = self.out_proj(y)
        y = self.drop(y)
        out = residual + y
        if return_features:
            return out, {
                "x_before_modulation": x_before_modulation.detach(),
                "x_modulated": x_modulated.detach(),
                "out_after_mamba":out.detach()
            }
        return out


class VisionMambaBlock(nn.Module):
    def __init__(self, dim, text_dim=None, drop_path=0., drop=0., norm_layer=nn.BatchNorm2d, *, use_visual_modulation, visual_modulation_mid_amp=0.05, visual_modulation_high_amp=0.10, scan_mode="bidi_snake"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mamba = VisionMambaMixer(dim=dim, text_dim=text_dim, drop=drop, use_visual_modulation=use_visual_modulation, visual_modulation_mid_amp=visual_modulation_mid_amp, visual_modulation_high_amp=visual_modulation_high_amp, scan_mode=scan_mode)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, 3 * dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(3 * dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, guidance, text_guidance=None, return_features=False):
        if return_features:
            mamba_out, features = self.mamba(self.norm1(x), guidance, text_guidance, return_features=True)
        else:
            mamba_out = self.mamba(self.norm1(x), guidance, text_guidance)
            features = None

        x = x + self.drop_path(mamba_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_features:
            return x, features
        return x


class SptialFusionBlock(nn.Module):
    def __init__(self, inter_channels, num_heads, drop_path=0., sr_ratio=1, text_dim=None, *, use_visual_modulation, visual_modulation_mid_amp=0.05, visual_modulation_high_amp=0.10, scan_mode="bidi_snake"):
        super(SptialFusionBlock, self).__init__()
        self.sa = VisionMambaBlock(inter_channels, text_dim=text_dim, drop_path=drop_path, use_visual_modulation=use_visual_modulation, visual_modulation_mid_amp=visual_modulation_mid_amp, visual_modulation_high_amp=visual_modulation_high_amp, scan_mode=scan_mode)
    
    def forward(self, x, img_guidance, text_guidance=None, return_features=False):
        spatial_feat = self.sa(x, img_guidance, text_guidance, return_features=return_features)
        return spatial_feat