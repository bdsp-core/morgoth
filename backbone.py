import math
from functools import partial
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path, to_2tuple, trunc_normal_
from timm.models import register_model
from einops import rearrange
from timm.layers import trunc_normal_ as __call_trunc_normal_
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')

def _morgoth_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class MorgothDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(MorgothDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class MorgothMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MorgothAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))

            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        attn_bias = None
        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn_bias = relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn_bias = rel_pos_bias if attn_bias is None else attn_bias + rel_pos_bias

        if return_attention or return_qkv or not _HAS_SDPA:
            q_scaled = q * self.scale
            attn = (q_scaled @ k.transpose(-2, -1))
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if return_attention:
                return attn

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

            if return_qkv:
                return x, qkv

            return x

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MorgothBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MorgothAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = MorgothDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MorgothMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    @staticmethod
    def _residual(x, sublayer_out, gamma, drop_path):
        """x + drop_path(gamma * sublayer_out), or x + drop_path(sublayer_out)
        when this block has no layer-scale gamma (init_values<=0, so no such
        parameter exists in this instance's state_dict at all). Pulled out
        of forward() below to remove the repeated if/else for the attention
        and MLP sublayers -- identical arithmetic to before, just written
        once instead of twice."""
        if gamma is None:
            return x + drop_path(sublayer_out)
        return x + drop_path(gamma * sublayer_out)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        x = self._residual(x, self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias), self.gamma_1, self.drop_path)
        x = self._residual(x, self.mlp(self.norm2(x)), self.gamma_2, self.drop_path)
        return x


class MorgothPatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MorgothTemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        B, N, A, T = x.shape
        x = x.reshape(B, N * A, T)
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        B2, C2, NA2, T2 = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B2, NA2, T2 * C2)
        return x


class MorgothEEGTransformer(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200,
                 depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, use_grad_checkpoint=False, **kwargs):
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = MorgothTemporalConv(out_chans=out_chans) if in_chans == 1 else MorgothPatchEmbed(EEG_size=EEG_size,
                                                                                              patch_size=patch_size,
                                                                                              in_chans=in_chans,
                                                                                              embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            MorgothBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _embed_tokens(self, x, input_chans=None, input_time_window=None, n_channels=None):
        """Prepend the cls token, then add the absolute position embedding
        (sliced to input_chans when given) and the per-patch time embedding.
        Shared by forward_features / forward_intermediate /
        get_intermediate_layers below -- pure code-organization refactor
        (was three copies of this block), identical arithmetic to before.
        Callers that omit input_time_window/n_channels get the same fixed
        self.time_window / 62-channel defaults those two methods always
        used; forward_features passes the input-derived values it always
        computed dynamically."""
        batch_size = x.shape[0]
        time_window = input_time_window if input_time_window is not None else self.time_window
        n_ch = n_channels if n_channels is not None else 62

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, n_ch, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        return self.pos_drop(x)

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        n_channels = n if t == self.patch_size else a
        x = self.patch_embed(x)
        x = self._embed_tokens(x, input_chans=input_chans, input_time_window=input_time_window, n_channels=n_channels)

        if self.use_grad_checkpoint and self.training:
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x, None, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        x = self._embed_tokens(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        x = self._embed_tokens(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features


@register_model
def morgoth_backbone_base(pretrained=False, **kwargs):
    model = MorgothEEGTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


@register_model
def morgoth_backbone_large(pretrained=False, **kwargs):
    model = MorgothEEGTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model





def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class MorgothTemporalConv(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        B, N, A, T = x.shape
        x = x.reshape(B, N * A, T)
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        B2, C2, NA2, T2 = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B2, NA2, T2 * C2)
        return x


class MorgothEEGTransformerForMaskedModeling(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = MorgothTemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MorgothBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans, bool_masked_pos):
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False,
                return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, input_chans=input_chans, bool_masked_pos=bool_masked_pos)
        if return_all_patch_tokens:
            return x
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            return self.lm_head(x[bool_masked_pos])

    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x)
            q, k, v = x.chunk(3, dim=-1)
            b, n, c = q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)


class MorgothEEGTransformerForMEM(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.student = MorgothEEGTransformerForMaskedModeling(EEG_size, patch_size, in_chans, out_chans, vocab_size,
                                                             embed_dim, depth,
                                                             num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale,
                                                             drop_rate, attn_drop_rate, drop_path_rate, norm_layer,
                                                             init_values, attn_head_dim,
                                                             use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias,
                                                             init_std)

        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        trunc_normal_(self.lm_head.weight, std=init_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}

    def forward(self, x, input_chans=None, bool_masked_pos=None):
        x_masked = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls = x_masked[:, 1:]
        x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos])

        x_masked_sym = self.student(x, input_chans, ~bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls_sym = x_masked_sym[:, 1:]
        x_rec_sym = self.lm_head(x_masked_no_cls_sym[~bool_masked_pos])

        return x_rec, x_rec_sym


@register_model
def morgoth_pretrain_base(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = MorgothEEGTransformerForMEM(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _morgoth_cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def morgoth_pretrain_large(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = MorgothEEGTransformerForMEM(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _morgoth_cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def morgoth_pretrain_huge(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = MorgothEEGTransformerForMEM(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _morgoth_cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def morgoth_backbone_huge(pretrained=False, **kwargs):
    model = MorgothEEGTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model



def _rope_build_freqs(head_dim: int, seq_len: int, base: float = 10000.0, device=None):
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def _rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(q, k, cos, sin):
    cos = cos[None, None, :, :].type_as(q)
    sin = sin[None, None, :, :].type_as(q)
    q_rot = q * cos + _rope_rotate_half(q) * sin
    k_rot = k * cos + _rope_rotate_half(k) * sin
    return q_rot, k_rot


class RotaryAttention(nn.Module):
    """Multi-head self-attention using rotary position embeddings instead
    of MorgothAttention's learned relative_position_bias_table -- a different
    position-encoding mechanism."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., rope_base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = _rope_build_freqs(self.head_dim, N, base=self.rope_base, device=x.device)
        q, k = _rope_apply(q, k, cos, sin)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope_base=10000.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RotaryAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rope_base=rope_base)
        self.drop_path = MorgothDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MorgothMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, **kwargs):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MorgothEEGTransformerRoPE(nn.Module):
    """RoPE counterpart of MorgothEEGTransformer. Same constructor keyword names
    where they still apply (so the same @register_model call-site pattern
    can build it), but carries no pos_embed / time_embed parameters --
    position information comes entirely from the rotary embeddings applied
    inside each RoPEBlock's attention. use_abs_pos_emb / use_rel_pos_bias /
    use_shared_rel_pos_bias are accepted for interface compatibility and
    ignored (RoPE replaces all three)."""

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200,
                 depth=12, num_heads=10, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_mean_pooling=True, init_scale=0.001, rope_base=10000.0, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = MorgothTemporalConv(out_chans=out_chans) if in_chans == 1 else MorgothPatchEmbed(
            EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            RoPEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, rope_base=rope_base)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x


@register_model
def morgoth_backbone_base_rope(pretrained=False, **kwargs):
    model = MorgothEEGTransformerRoPE(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


@register_model
def morgoth_backbone_large_rope(pretrained=False, **kwargs):
    model = MorgothEEGTransformerRoPE(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model



class MorgothSwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer, "GLU Variants Improve Transformer",
    2020 -- the gated-MLP used by LLaMA/PaLM). fc1 projects to 2x hidden,
    splits into (gate, value), applies silu(gate) * value, then fc2 projects
    back down. A genuinely different gating mechanism from MorgothMLP's
    plain two-layer GELU MLP."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MorgothMoEFFN(nn.Module):
    """Sparse Mixture-of-Experts feed-forward (Switch-Transformer/Mixtral-
    style top-k token routing). A small linear router scores `num_experts`
    independent MLP experts per token; only the top_k highest-scoring
    experts actually run for each token, and their outputs are combined
    weighted by the (renormalized) router probability. This scales model
    capacity (total expert parameters) largely independently of per-token
    compute (only top_k experts run), a fundamentally different mechanism
    from MorgothMLP's single dense MLP applied identically to every token."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 num_experts=4, top_k=1, act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.out_features = out_features
        self.router = nn.Linear(in_features, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                act_layer(),
                nn.Linear(hidden_features, out_features),
            ) for _ in range(num_experts)
        ])
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        x_flat = x.reshape(-1, C)

        router_probs = F.softmax(self.router(x_flat), dim=-1)
        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = x_flat.new_zeros(x_flat.shape[0], self.out_features)
        for e_idx, expert in enumerate(self.experts):
            token_mask = (topk_idx == e_idx).any(dim=-1)
            if not token_mask.any():
                continue
            expert_out = expert(x_flat[token_mask])
            weight = torch.where(
                topk_idx[token_mask] == e_idx, topk_probs[token_mask],
                torch.zeros_like(topk_probs[token_mask])
            ).sum(dim=-1, keepdim=True)
            out[token_mask] = out[token_mask] + expert_out * weight

        out = self.drop(out)
        return out.reshape(B, N, self.out_features)


class MorgothChannelGraphAttention(nn.Module):
    """Attention applied across the EEG-channel axis specifically, instead
    of the fully-flattened channel*patch token sequence the main
    MorgothBlock stack operates on. Each electrode's own pooled
    representation is treated as one graph node; the softmax attention
    weights act as a learned, dynamic, fully-connected adjacency matrix
    over electrodes (rather than a fixed montage-distance graph). Meant to
    be applied once, after per-channel mean-pooling of the main backbone's
    patch tokens, as an additional channel-relationship signal."""

    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, channel_tokens):
        x = self.norm(channel_tokens)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return channel_tokens + out


class MorgothExtendedBlock(nn.Module):
    """Same as MorgothBlock, but the feed-forward sublayer is selectable:
    'mlp' (default, identical to MorgothBlock), 'swiglu' (MorgothSwiGLU), or
    'moe' (MorgothMoEFFN)."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None,
                 ffn_type='mlp', moe_num_experts=4, moe_top_k=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MorgothAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = MorgothDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if ffn_type == 'swiglu':
            self.mlp = MorgothSwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif ffn_type == 'moe':
            self.mlp = MorgothMoEFFN(in_features=dim, hidden_features=mlp_hidden_dim,
                                     num_experts=moe_num_experts, top_k=moe_top_k,
                                     act_layer=act_layer, drop=drop)
        else:
            self.mlp = MorgothMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, **kwargs):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MorgothEEGTransformerExtended(nn.Module):
    """Same overall design as MorgothEEGTransformer (TemporalConv/PatchEmbed
    stem, cls token, learned abs pos_embed + time_embed, stack of blocks,
    mean-pool or cls-token head), plus three independently-toggleable
    additions, all OFF by default:
      - ffn_type='swiglu' | 'moe' (default 'mlp' = identical to MorgothBlock)
      - use_channel_graph_attn=True adds one MorgothChannelGraphAttention
        layer over per-channel-pooled patch tokens, whose pooled output is
        averaged into the final feature before the head (default False =
        no extra layer, no extra parameters)
    With every flag at its default, this class is architecturally identical
    to MorgothEEGTransformer (same parameter count/shapes) -- it is kept as
    a separate registered model rather than changing MorgothEEGTransformer
    itself, so no existing checkpoint or script is affected either way."""

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200,
                 depth=12, num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_mean_pooling=True, init_scale=0.001,
                 ffn_type='mlp', moe_num_experts=4, moe_top_k=1,
                 use_channel_graph_attn=False, graph_attn_heads=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = MorgothTemporalConv(out_chans=out_chans) if in_chans == 1 else MorgothPatchEmbed(
            EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MorgothExtendedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values, window_size=None,
                ffn_type=ffn_type, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.use_channel_graph_attn = use_channel_graph_attn
        if use_channel_graph_attn:
            self.channel_graph_attn = MorgothChannelGraphAttention(embed_dim, num_heads=graph_attn_heads)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer.mlp, 'fc2'):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)

        if self.fc_norm is not None:
            pooled_patches = x[:, 1:, :]
            if self.use_channel_graph_attn:
                channel_tokens = pooled_patches.reshape(batch_size, n, -1, pooled_patches.shape[-1]).mean(dim=2)
                channel_tokens = self.channel_graph_attn(channel_tokens)
                graph_feat = channel_tokens.mean(dim=1)
                base_feat = self.fc_norm(pooled_patches.mean(1))
                pooled = 0.5 * base_feat + 0.5 * graph_feat
            else:
                pooled = self.fc_norm(pooled_patches.mean(1))

            if return_all_tokens:
                return self.fc_norm(x)
            if return_patch_tokens:
                return self.fc_norm(pooled_patches)
            return pooled
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x


@register_model
def morgoth_backbone_base_swiglu(pretrained=False, **kwargs):
    kwargs.pop('ffn_type', None)
    model = MorgothEEGTransformerExtended(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), ffn_type='swiglu', **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


@register_model
def morgoth_backbone_base_moe(pretrained=False, moe_num_experts=4, moe_top_k=1, **kwargs):
    kwargs.pop('ffn_type', None)
    model = MorgothEEGTransformerExtended(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), ffn_type='moe',
        moe_num_experts=moe_num_experts, moe_top_k=moe_top_k, **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


@register_model
def morgoth_backbone_base_graph(pretrained=False, graph_attn_heads=4, **kwargs):
    kwargs.pop('use_channel_graph_attn', None)
    model = MorgothEEGTransformerExtended(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_channel_graph_attn=True,
        graph_attn_heads=graph_attn_heads, **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model



def _morgoth_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Standard ALiBi per-head slope schedule (Press, Smith & Lewis,
    "Train Short, Test Long: Attention with Linear Biases Enables Input
    Length Extrapolation", 2021). Geometric sequence of slopes so different
    heads penalize distance at different rates."""
    def slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = slopes_power_of_2(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        slopes = slopes_power_of_2(closest)
        extra = _morgoth_alibi_slopes(2 * closest)[0::2][: num_heads - closest]
        slopes = slopes + extra.tolist()
    return torch.tensor(slopes, dtype=torch.float32)


class MorgothAlibiAttention(nn.Module):
    """Multi-head self-attention with ALiBi (linear distance bias) instead
    of MorgothAttention's learned relative_position_bias_table or
    RotaryAttention's rotary embeddings -- a third, distinct position-
    encoding mechanism: a fixed (non-learned), per-head-scaled linear
    penalty on |query_pos - key_pos| added directly to the attention
    logits before softmax."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.register_buffer('alibi_slopes', _morgoth_alibi_slopes(num_heads), persistent=False)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        pos = torch.arange(N, device=x.device)
        rel_dist = (pos[None, :] - pos[:, None]).abs().to(attn.dtype)
        bias = -self.alibi_slopes.to(attn.device).view(1, -1, 1, 1) * rel_dist.view(1, 1, N, N)
        attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MorgothAlibiBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MorgothAlibiAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = MorgothDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MorgothMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, **kwargs):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MorgothEEGTransformerAlibi(nn.Module):
    """ALiBi counterpart of MorgothEEGTransformer. Same shape as
    MorgothEEGTransformerRoPE (no pos_embed/time_embed parameters --
    position information comes entirely from the ALiBi bias inside each
    attention layer)."""

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200,
                 depth=12, num_heads=10, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_mean_pooling=True, init_scale=0.001, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = MorgothTemporalConv(out_chans=out_chans) if in_chans == 1 else MorgothPatchEmbed(
            EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MorgothAlibiBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                              norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        return self.head(x)


@register_model
def morgoth_backbone_base_alibi(pretrained=False, **kwargs):
    model = MorgothEEGTransformerAlibi(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


class MorgothHierarchicalPatchEmbed(nn.Module):
    """Multi-scale temporal stem: two parallel MorgothTemporalConv-style
    branches at different temporal granularities (a 'fine' branch matching
    the original stride-8 stem, and a 'coarse' branch with roughly double
    the receptive field / stride), concatenated per-patch and projected
    back to embed_dim. Gives the token sequence access to two temporal
    resolutions instead of the single fixed patch granularity the original
    MorgothTemporalConv/MorgothPatchEmbed produce."""

    def __init__(self, out_chans=8, embed_dim=200):
        super().__init__()
        self.fine = MorgothTemporalConv(out_chans=out_chans)
        self.coarse_conv1 = nn.Conv2d(1, out_chans, kernel_size=(1, 31), stride=(1, 8), padding=(0, 15))
        self.coarse_gelu1 = nn.GELU()
        self.coarse_norm1 = nn.GroupNorm(4, out_chans)
        self.coarse_conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.coarse_gelu2 = nn.GELU()
        self.coarse_norm2 = nn.GroupNorm(4, out_chans)
        self.merge = nn.Linear(2 * out_chans * 25, embed_dim)

    def _coarse(self, x):
        B, N, A, T = x.shape
        x = x.reshape(B, N * A, T).unsqueeze(1)
        x = self.coarse_gelu1(self.coarse_norm1(self.coarse_conv1(x)))
        x = self.coarse_gelu2(self.coarse_norm2(self.coarse_conv2(x)))
        B2, C2, NA2, T2 = x.shape
        return x.permute(0, 2, 3, 1).reshape(B2, NA2, T2 * C2)

    def forward(self, x, **kwargs):
        fine_tokens = self.fine(x)
        coarse_tokens = self._coarse(x)
        merged = torch.cat([fine_tokens, coarse_tokens], dim=-1)
        return self.merge(merged)


class MorgothEEGTransformerHierarchical(MorgothEEGTransformer):
    """MorgothEEGTransformer with MorgothHierarchicalPatchEmbed swapped in
    for the stem. Subclasses MorgothEEGTransformer (reusing all of its
    block-stack / pos_embed / head / forward machinery unchanged) and only
    replaces self.patch_embed after the parent constructor runs, plus the
    dependent merge projection's out_chans*25 assumption (patch_size=200,
    stride 8 -> 25 timesteps per patch after the temporal conv stem)."""

    def __init__(self, *args, out_chans=8, embed_dim=200, **kwargs):
        super().__init__(*args, out_chans=out_chans, embed_dim=embed_dim, **kwargs)
        self.patch_embed = MorgothHierarchicalPatchEmbed(out_chans=out_chans, embed_dim=embed_dim)


@register_model
def morgoth_backbone_base_hierarchical(pretrained=False, **kwargs):
    model = MorgothEEGTransformerHierarchical(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


class MorgothSSMBlock(nn.Module):
    """Simplified selective state-space sequence mixer (Mamba-style, Gu &
    Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces", 2023) used in place of self-attention: a per-channel linear
    recurrence h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t,
    y_t = C_t . h_t, where delta/B/C are all computed from the input itself
    (input-dependent / "selective", unlike a fixed-A linear SSM). This is a
    fundamentally different sequence-mixing mechanism from attention -- no
    O(N^2) all-pairs attention matrix, an O(N) sequential recurrence
    instead. Reference implementation (a plain per-step Python loop) for
    correctness, not a reimplementation of the official fused/parallel-scan
    CUDA kernel's speed."""

    def __init__(self, dim, state_dim=16, expand=2, drop=0.):
        super().__init__()
        inner_dim = dim * expand
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.x_proj = nn.Linear(inner_dim, state_dim * 2 + 1)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
                                  .unsqueeze(0).repeat(inner_dim, 1))
        self.out_proj = nn.Linear(inner_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, _ = x.shape
        x_in, gate = self.in_proj(x).chunk(2, dim=-1)
        x_act = F.silu(x_in)

        Bp, Cp, delta = self.x_proj(x_act).split([self.state_dim, self.state_dim, 1], dim=-1)
        delta = F.softplus(delta)
        A = -torch.exp(self.A_log)

        h = x.new_zeros(B, self.inner_dim, self.state_dim)
        ys = []
        for t in range(N):
            dt = delta[:, t]
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dBx = dt.unsqueeze(-1) * Bp[:, t].unsqueeze(1) * x_act[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cp[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) * F.silu(gate)
        return self.drop(self.out_proj(y))


class MorgothSSMLayer(nn.Module):
    def __init__(self, dim, state_dim=16, expand=2, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ssm = MorgothSSMBlock(dim, state_dim=state_dim, expand=expand, drop=drop)
        self.drop_path = MorgothDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MorgothMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, **kwargs):
        x = x + self.drop_path(self.ssm(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MorgothEEGTransformerSSM(nn.Module):
    """Selective-state-space counterpart of MorgothEEGTransformer: same
    stem/cls-token/pos+time-embedding/pooling scaffolding, but each
    MorgothBlock is replaced by a MorgothSSMLayer (SSM sequence mixer + MLP,
    no self-attention at all)."""

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200,
                 depth=12, mlp_ratio=4., drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 use_mean_pooling=True, init_scale=0.001, state_dim=16, ssm_expand=2,
                 use_abs_pos_emb=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = MorgothTemporalConv(out_chans=out_chans) if in_chans == 1 else MorgothPatchEmbed(
            EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MorgothSSMLayer(embed_dim, state_dim=state_dim, expand=ssm_expand, mlp_ratio=mlp_ratio,
                            drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        n_channels = n if t == self.patch_size else a
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(
                batch_size, n_channels, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t_ = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t_)
            return self.fc_norm(t_.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        return self.head(x)


@register_model
def morgoth_backbone_base_ssm(pretrained=False, **kwargs):
    model = MorgothEEGTransformerSSM(
        patch_size=200, embed_dim=200, depth=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model



class MorgothEEGTransformerMAE(nn.Module):
    """MAE-style pretraining ("Masked Autoencoders Are Scalable Vision
    Learners", He et al. 2021): reconstructs the RAW signal of masked
    patches directly with an MSE loss, instead of MorgothEEGTransformerForMEM's
    objective of classifying each masked patch's VQ codebook index (cross-
    entropy against a frozen tokenizer's labels). This is the core BEiT-vs-MAE
    paradigm difference (regress the signal vs. classify a discrete token).

    Reuses MorgothEEGTransformerForMaskedModeling as the encoder (mask_token
    substitution, full-sequence processing -- for simplicity/reuse, not the
    compute-saving "encoder only sees visible tokens" trick from the
    original MAE paper), then a small decoder (a couple of transformer
    blocks) projects the encoder output back to raw per-patch signal
    (patch_size samples)."""

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=None,
                 use_abs_pos_emb=True, decoder_depth=2, **kwargs):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.encoder = MorgothEEGTransformerForMaskedModeling(
            EEG_size, patch_size, in_chans, out_chans, 8192, embed_dim, depth, num_heads, mlp_ratio,
            qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer,
            init_values, None, use_abs_pos_emb, False, False, 0.02)
        self.decoder_blocks = nn.ModuleList([
            MorgothBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        norm_layer=norm_layer, init_values=init_values)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size)

    def forward(self, x, input_chans=None, bool_masked_pos=None):
        """x: [B, N_channels, N_patches, patch_size]. Returns
        (pred_masked, target_masked) -- caller computes F.mse_loss on these,
        the direct-signal-reconstruction analogue of MorgothEEGTransformerForMEM's
        (x_rec, x_rec_sym) cross-entropy pair."""
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(
                (x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool, device=x.device)

        enc = self.encoder(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
        for blk in self.decoder_blocks:
            enc = blk(enc)
        enc = self.decoder_norm(enc)
        pred = self.decoder_pred(enc[:, 1:])

        B, N, A, T = x.shape
        target = x.reshape(B, N * A, T)
        return pred[bool_masked_pos], target[bool_masked_pos]


@register_model
def morgoth_pretrain_base_mae(pretrained=False, **kwargs):
    kwargs.pop('vocab_size', None)
    model = MorgothEEGTransformerMAE(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6), norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


class MorgothEEGTransformerForMEMContrastive(MorgothEEGTransformerForMEM):
    """MorgothEEGTransformerForMEM plus an NT-Xent/InfoNCE contrastive term
    computed from two augmented views of the same input batch. Adds ZERO new
    parameters -- MorgothEEGTransformerForMEM already defines a
    `projection_head` that its own forward() never actually uses; this
    subclass is the first thing to exercise it. Fully opt-in: forward()
    behaves exactly like the parent class unless x_aug is explicitly passed."""

    @staticmethod
    def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1):
        B = z1.shape[0]
        all_z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(all_z, all_z.t()) / temperature
        sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z1.device), float('-inf'))
        targets = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(B, device=z1.device),
        ])
        return F.cross_entropy(sim, targets)

    def _pooled_projection(self, x, input_chans, bool_masked_pos):
        feat = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
        pooled = feat.mean(dim=1)
        return F.normalize(self.projection_head(pooled), dim=-1)

    def forward(self, x, input_chans=None, bool_masked_pos=None, x_aug=None, contrastive_temperature=0.1):
        x_rec, x_rec_sym = super().forward(x, input_chans, bool_masked_pos)
        cont_loss = None
        if x_aug is not None:
            z1 = self._pooled_projection(x, input_chans, bool_masked_pos)
            z2 = self._pooled_projection(x_aug, input_chans, bool_masked_pos)
            cont_loss = self.nt_xent(z1, z2, contrastive_temperature)
        return x_rec, x_rec_sym, cont_loss


@register_model
def morgoth_pretrain_base_contrastive(pretrained=False, **kwargs):
    if 'vocab_size' in kwargs:
        vocab_size = kwargs.pop('vocab_size')
    else:
        vocab_size = 8192
    model = MorgothEEGTransformerForMEMContrastive(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6), norm_layer=partial(nn.LayerNorm, eps=1e-6),
        vocab_size=vocab_size, **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model


class MorgothCausalAttention(nn.Module):
    """Same computation as MorgothAttention, but with a causal mask: each
    position can only attend to itself and earlier positions in the
    sequence. Used by MorgothEEGTransformerCausal for GPT-style
    autoregressive pretraining instead of BEiT-style bidirectional masked
    modeling."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if _HAS_SDPA:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale,
                dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            causal_mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class MorgothCausalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MorgothCausalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = MorgothDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MorgothMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, **kwargs):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MorgothEEGTransformerCausal(nn.Module):
    """GPT-style autoregressive pretraining model: causal self-attention
    over the flattened (channel, patch) sequence; the representation at
    each position predicts the frozen tokenizer's VQ codebook index of the
    NEXT patch in sequence order (same label source as
    MorgothEEGTransformerForMEM, just a next-token objective instead of a
    random-position masked one). A fundamentally different pretraining
    paradigm from the existing masked modeling (GPT/unidirectional vs
    BERT-BEiT/bidirectional) -- there is no mask ratio and no mask_token at
    all here, every position is always visible to its causal predecessors."""

    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200,
                 depth=12, num_heads=10, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, **kwargs):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = MorgothTemporalConv(out_chans=out_chans)
        self.patch_size = patch_size
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MorgothCausalBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                              norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.lm_head.weight, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, input_chans=None):
        """x: [B, N_channels, N_patches, patch_size]. Returns logits
        [B, N_channels*N_patches, vocab_size]; caller computes
        cross_entropy(logits[:, :-1], vq_token_ids[:, 1:]) for next-patch
        prediction."""
        batch_size, n, a, t = x.shape
        x = self.patch_embed(x)
        time_embed = self.time_embed[:, 0:a, :].unsqueeze(1).expand(batch_size, n, -1, -1).flatten(1, 2)
        x = x + time_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.lm_head(x)


@register_model
def morgoth_pretrain_base_causal(pretrained=False, **kwargs):
    if 'vocab_size' in kwargs:
        vocab_size = kwargs.pop('vocab_size')
    else:
        vocab_size = 8192
    model = MorgothEEGTransformerCausal(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _morgoth_cfg()
    return model
