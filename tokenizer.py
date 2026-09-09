import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.layers import trunc_normal_
from timm.models import register_model
import torch.distributed as distributed
from einops import rearrange, repeat
from backbone import MorgothEEGTransformer


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        if codebook_init_path == '':
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))


class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5,
                 statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay

        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)

        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)

    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c')
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flattened)

        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)

        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)

        if self.training and self.embedding.update:

            bins = encodings.sum(0)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)

            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)

            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        loss = self.beta * F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()

        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices



class MorgothVQTokenizer(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192,
                 embed_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=200,
                 smooth_l1_loss=False,
                 use_contrastive=False,
                 contrastive_proj_dim=128,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        print('Final encoder config', encoder_config)
        self.encoder = MorgothEEGTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = MorgothEEGTransformer(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['EEG_size'] // self.patch_size)

        self.decoder_out_dim = decoder_out_dim

        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim)
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        self.use_contrastive = use_contrastive
        if self.use_contrastive:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
                nn.GELU(),
                nn.Linear(encoder_config['embed_dim'], contrastive_proj_dim),
            )
            self.contrastive_proj.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')

        return output

    def encode(self, x, input_chans=None):
        batch_size, n, a, t = x.shape
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h,
                                          w=w)
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss

    def decode(self, quantize, input_chans=None, **kwargs):
        decoder_features = self.decoder(quantize, input_chans, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)
        rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec, rec_angle

    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        return self.get_tokens(x, input_chans, **kwargs)['token']

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def _contrastive_embed(self, x, input_chans=None):
        feat = self.encoder(x, input_chans, return_patch_tokens=True)
        return l2norm(self.contrastive_proj(feat.mean(dim=1)))

    @staticmethod
    def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1):
        """SimCLR-style NT-Xent / InfoNCE: anchor=z1, positive=z2 (same
        sample's other augmented view), negatives = every other sample in
        the batch (both views)."""
        B = z1.shape[0]
        all_z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(all_z, all_z.t()) / temperature
        sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z1.device), float('-inf'))
        targets = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(B, device=z1.device),
        ])
        return F.cross_entropy(sim, targets)

    def forward(self, x, input_chans=None, x_aug=None,
                contrastive_weight: float = 0.0, contrastive_temperature: float = 0.1,
                **kwargs):
        """
        x: shape [B, N, T]
        x_aug: optional second augmented view of the same batch, shape [B, N, T].
            Only used when this model was constructed with use_contrastive=True
            AND contrastive_weight > 0 -- otherwise ignored, so this stays a
            no-op for every existing caller.
        """

        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)

        quantize, embed_ind, emb_loss = self.encode(x, input_chans)

        xrec, xrec_angle = self.decode(quantize, input_chans)
        rec_loss = self.calculate_rec_loss(xrec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(xrec_angle, angle)
        loss = emb_loss + rec_loss + rec_angle_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        log[f'{split}/rec_angle_loss'] = rec_angle_loss.detach().mean()

        if self.use_contrastive and contrastive_weight > 0.0 and x_aug is not None:
            x_aug = rearrange(x_aug, 'B N (A T) -> B N A T', T=200)
            with torch.cuda.amp.autocast(enabled=False):
                z1 = self._contrastive_embed(
                    x.type_as(self.contrastive_proj[-1].weight), input_chans)
                z2 = self._contrastive_embed(
                    x_aug.type_as(self.contrastive_proj[-1].weight), input_chans)
            cont_loss = self.nt_xent(z1, z2, contrastive_temperature)
            loss = loss + contrastive_weight * cont_loss
            log[f'{split}/contrastive_loss'] = cont_loss.detach().mean()

        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log


def morgoth_tokenizer_default_params():
    return dict(EEG_size=1600, patch_size=200, in_chans=1, num_classes=1000, embed_dim=200, depth=12, num_heads=10,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True,
                use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


def _load_tokenizer_pretrained_weights(model, pretrained, pretrained_weight):
    """Shared `as_tokenzer=True` weight-loading logic used by every
    registered tokenizer factory below (was copy-pasted 3x: base, large,
    and the FSQ variant). Pure code-organization refactor -- identical
    logic/order to what each factory did inline before."""
    assert pretrained
    assert pretrained_weight is not None

    if pretrained_weight.startswith('https'):
        weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
    else:
        weights = torch.load(pretrained_weight, map_location='cpu')

    if 'model' in weights:
        weights = weights['model']
    else:
        weights = weights["state_dict"]
    keys = list(weights.keys())

    for k in keys:
        if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
            del weights[k]
    model.load_state_dict(weights)
    return model


@register_model
def morgoth_tokenizer_base(pretrained=False, pretrained_weight=None, as_tokenzer=False, EEG_size=1600,
                                        n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = morgoth_tokenizer_default_params(), morgoth_tokenizer_default_params()

    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = MorgothVQTokenizer(encoder_config, decoder_config, n_code, code_dim,
                  decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        model = _load_tokenizer_pretrained_weights(model, pretrained, pretrained_weight)
    return model


@register_model
def morgoth_tokenizer_large(pretrained=False, pretrained_weight=None, as_tokenzer=False, EEG_size=1600,
                                         n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = morgoth_tokenizer_default_params(), morgoth_tokenizer_default_params()

    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    encoder_config['depth'] = 24
    decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = MorgothVQTokenizer(encoder_config, decoder_config, n_code, code_dim,
                  decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        model = _load_tokenizer_pretrained_weights(model, pretrained, pretrained_weight)
    return model



def _fsq_round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient estimator."""
    z_rounded = torch.round(z)
    return z + (z_rounded - z).detach()


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantizer. Same call signature as
    NormEMAVectorQuantizer.forward(z) -> (quantized, loss, indices), so it
    is a drop-in replacement for MorgothVQTokenizerFSQ, but the mechanism is entirely
    different: independent per-channel scalar rounding instead of
    nearest-neighbor lookup against a learned codebook."""

    def __init__(self, levels=(8, 8, 8, 6, 5)):
        super().__init__()
        levels = list(levels)
        self.levels = levels
        self.dim = len(levels)
        self.n_e = 1
        for lv in levels:
            self.n_e *= lv
        self.register_buffer('_levels_t', torch.tensor(levels, dtype=torch.float32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer('_basis', basis, persistent=False)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        half_l = (self._levels_t - 1) * 0.5
        return torch.tanh(z) * half_l

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        bounded = self._bound(z)
        rounded = _fsq_round_ste(bounded)
        half_l = (self._levels_t - 1) * 0.5
        return rounded / half_l.clamp(min=1e-6)

    def _codes_to_indices(self, zhat_norm: torch.Tensor) -> torch.Tensor:
        half_l = (self._levels_t - 1) * 0.5
        zhat_int = torch.round(zhat_norm * half_l + half_l)
        return (zhat_int * self._basis).sum(dim=-1).long()

    def forward(self, z: torch.Tensor):
        z = rearrange(z, 'b c h w -> b h w c')
        assert z.shape[-1] == self.dim, (
            f'FSQQuantizer expects last dim == len(levels) == {self.dim}, got {z.shape[-1]}. '
            f'Set code_dim=len(fsq_levels) when building MorgothVQTokenizerFSQ.')
        zq = self._quantize(z)
        indices = self._codes_to_indices(zq)
        zq = rearrange(zq, 'b h w c -> b c h w')
        loss = z.new_zeros(())
        return zq, loss, indices


class MorgothVQTokenizerFSQ(MorgothVQTokenizer):
    """MorgothVQTokenizer with its quantizer swapped for FSQQuantizer. Subclasses MorgothVQTokenizer
    (reusing its encoder/decoder/task-layer/forward machinery unchanged) and
    only replaces self.quantize after the parent constructor runs, so the
    only architectural difference from MorgothVQTokenizer is the quantization mechanism
    itself."""

    def __init__(self, encoder_config, decoder_config, fsq_levels=(8, 8, 8, 6, 5),
                 decoder_out_dim=200, smooth_l1_loss=False, **kwargs):
        embed_dim = len(fsq_levels)
        super().__init__(
            encoder_config, decoder_config, n_embed=1, embed_dim=embed_dim,
            decoder_out_dim=decoder_out_dim, smooth_l1_loss=smooth_l1_loss, **kwargs)
        self.quantize = FSQQuantizer(levels=fsq_levels)

    def get_number_of_tokens(self):
        return self.quantize.n_e


@register_model
def morgoth_tokenizer_base_fsq(pretrained=False, pretrained_weight=None, as_tokenzer=False,
                                             EEG_size=1600, fsq_levels=(8, 8, 8, 6, 5), **kwargs):
    encoder_config, decoder_config = morgoth_tokenizer_default_params(), morgoth_tokenizer_default_params()

    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = len(fsq_levels)
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = MorgothVQTokenizerFSQ(encoder_config, decoder_config, fsq_levels=fsq_levels,
                      decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        model = _load_tokenizer_pretrained_weights(model, pretrained, pretrained_weight)
    return model


if __name__ == '__main__':
    pass
