import math
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from einops import rearrange, reduce, repeat
from ema_pytorch import EMA
from skimage.metrics import structural_similarity
from torch import einsum, nn
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

from dataset import DIRD_Dataset
from flow_utils import flow_to_image, flow_warp, norm_flow, unnorm_flow, upsample2d_flow_as
from metric import torch_psnr as psnr

ssim = partial(structural_similarity, data_range=1.0, channel_axis=0)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

# constants

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


# helpers functions
def divisible_by(numer, denom):
    return (numer % denom) == 0


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# classifier free guidance functions


def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2),
            )
            if exists(time_emb_dim) or exists(classes_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.block3 = Block(dim * 2, dim, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None, condition=None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        x = torch.cat((x, condition), dim=1)
        x = self.block3(x)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# model


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob=0.5,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        attn_dim_head=32,
        attn_heads=4,
    ):
        super().__init__()

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)

        self.init_conv = nn.Conv2d(2, 3, 7, padding=3)
        self.init_conv1 = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        self.init_conv2 = nn.Conv2d(input_channels + 1, init_dim, 7, padding=3)  # XXX

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(nn.Linear(dim, classes_dim), nn.GELU(), nn.Linear(classes_dim, classes_dim))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                        ),
                        block_klass(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                        ),
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim - 1, 1)

    def forward_with_cond_scale(self, *args, cond_scale=1.0, rescaled_phi=0.0, **kwargs):
        logits = self.forward(*args, cond_drop_prob=0.0, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.0:
            return scaled_logits

        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)

    def forward(self, x, time, classes, condition, cond_drop_prob=None):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance

        classes_emb = self.classes_emb(classes)  # (bs,1)-(bs,64)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, "d -> b d", b=batch)

            classes_emb = torch.where(rearrange(keep_mask, "b -> b 1"), classes_emb, null_classes_emb)

        c = self.classes_mlp(classes_emb)  # (bs,64)-(bs,256)

        # unet

        x = self.init_conv(x)
        x = self.init_conv1(x)
        condition = self.init_conv2(condition)

        r = x.clone()
        r_c = condition.clone()

        t = self.time_mlp(time)

        h = []
        con = []
        for (
            block1,
            block2,
            attn,
            downsample,
        ) in self.downs:
            x = block1(x, t, c, condition)
            h.append(x)
            con.append(condition)

            x = block2(x, t, c, condition)
            x = attn(x)
            h.append(x)
            con.append(condition)

            x = downsample(x)
            condition = downsample(condition)

        x = self.mid_block1(x, t, c, condition)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c, condition)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            condition1 = torch.cat((condition, con.pop()), dim=1)
            x = block1(x, t, c, condition1)

            x = torch.cat((x, h.pop()), dim=1)
            condition2 = torch.cat((condition, con.pop()), dim=1)
            x = block2(x, t, c, condition2)
            x = attn(x)

            x = upsample(x)
            condition = upsample(condition)

        x = torch.cat((x, r), dim=1)
        condition = torch.cat((condition, r_c), dim=1)

        x = self.final_res_block(x, t, c, condition)
        return self.final_conv(x)


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        beta_schedule="cosine",
        ddim_sampling_eta=1.0,
        offset_noise_strength=0.0,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr
        elif objective == "pred_x0":
            loss_weight = maybe_clipped_snr
        elif objective == "pred_v":
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start

    def predict_start_from_v(self, x_t, t, v):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x,
        t,
        classes,
        condition,
        cond_scale=6.0,
        rescaled_phi=0.7,
        clip_x_start=False,
    ):
        model_output = self.model.forward_with_cond_scale(x, t, classes, condition, cond_scale=cond_scale, rescaled_phi=rescaled_phi)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised=True):
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale=6.0, rescaled_phi=0.7, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            classes=classes,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale=6.0, rescaled_phi=0.7):
        img = torch.randn(shape, device=self.betas.device)

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
            dynamic_ncols=True,
            leave=False,
        ):
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(
        self,
        classes,
        condition,
        shape,
        cond_scale=1.0,
        rescaled_phi=0,
        clip_denoised=True,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step", dynamic_ncols=True, leave=False):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                classes,
                condition,
                cond_scale=cond_scale,
                clip_x_start=clip_denoised,
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = unnorm_flow(img)
        return img

    @torch.no_grad()
    def sample(self, classes, condition, cond_scale=6.0, rescaled_phi=0.7):
        condition = normalize_to_neg_one_to_one(condition)
        batch_size, image_size, channels = (
            classes.shape[0],
            self.image_size,
            self.channels,
        )
        image_h, image_w = image_size
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            classes,
            condition,
            (batch_size, channels - 1, image_h, image_w),
            cond_scale,
            rescaled_phi,
        )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, condition, t, gt, *, classes, noise=None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model(x, t, classes, condition)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)

        pred_xstart = unnorm_flow(model_out)

        condition = unnormalize_to_zero_to_one(condition)[:, :3]
        warp_rs = flow_warp(condition, pred_xstart, pad="zeros", mode="bilinear")
        photo_loss = F.mse_loss(gt, warp_rs, reduction="none")
        photo_loss = reduce(photo_loss, "b ... -> b (...)", "mean")

        loss_total = loss.sum() + photo_loss.sum() / (photo_loss.sum() / loss.sum()).detach()

        return loss_total

    def forward(self, img, condition, gt, *args, **kwargs):
        b, _c, h, w, device, img_size = (*img.shape, img.device, self.image_size)
        img_h, img_w = self.image_size
        assert h == img_h and w == img_w, f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        condition = normalize_to_neg_one_to_one(condition)
        return self.p_losses(img, condition, t, gt, *args, **kwargs)


# trainer class


def get_light(flow):
    dx, dy = flow[:, 0], flow[:, 1]
    d = torch.sqrt(dx * dx + dy * dy)
    d = d / torch.max(d)
    return d


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        test_folder,
        real_image_size,
        *,
        train_batch_size=2,
        val_batch_size=2,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=False,
        max_grad_norm=1.0,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
            kwargs_handlers=[ddp_kwargs],
            project_config=ProjectConfiguration(project_dir=".", logging_dir=results_folder),
            log_with="tensorboard",
        )
        self.accelerator.init_trackers("result")

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert math.sqrt(num_samples) ** 2 == num_samples, "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = DIRD_Dataset(folder, real_image_size, self.image_size)
        self.test_ds = DIRD_Dataset(test_folder, real_image_size, self.image_size)

        assert len(self.ds) > 0, "Empty dataset detected."

        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cpu_count() // 2,
            persistent_workers=True,
        )
        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cpu_count() // 2,
            persistent_workers=True,
        )  # XXX: test is run on a single GPU
        self.test_long = len(self.test_dl)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def sample(self, save_root, device):
        self.ema.ema_model.eval()
        with torch.inference_mode():
            ssims = []
            psnrs = []
            for idx, test_data in tqdm(enumerate(self.test_dl), total=len(self.test_dl)):
                # [flow, downsampled_cond, downsampled_gt, cond, gt]
                flow = test_data[0].to(device)
                rs = test_data[1].to(device)

                show_rs = test_data[3].to(device)[:, :3]
                mask = test_data[3].to(device)[:, 3:]
                show_gs = test_data[4].to(device)
                names = test_data[5]

                image_classes = torch.zeros([rs.shape[0]]).to(device).to(torch.int64)
                out_flow = self.ema.ema_model.sample(classes=image_classes, condition=rs, cond_scale=1)
                flow_warp_show1 = upsample2d_flow_as(out_flow, show_rs, mode="bilinear", if_rate=True)
                flow_warp_show2 = upsample2d_flow_as(flow, show_rs, mode="bilinear", if_rate=True)

                all_images = flow_warp(show_rs, flow_warp_show1, pad="zeros", mode="bilinear")
                wm = flow_warp(mask, flow_warp_show1, pad="zeros", mode="bilinear")
                light = get_light(flow_warp_show1)
                _l = all_images.shape[0]

                for i in range(all_images.shape[0]):
                    name = names[i]
                    name = name.split(".")[0]

                    show_rs1 = show_rs[i]
                    wm1 = wm[i]
                    wm1 = 1.0 - wm1
                    light1 = light[i]
                    show_gs1 = show_gs[i].to(device)
                    pred_gs1 = all_images[i]
                    matrices = [show_rs1, show_gs1, pred_gs1]
                    result_matrix = torch.cat(matrices, axis=2)
                    ssims.append(ssim(show_gs1.cpu().numpy(), pred_gs1.cpu().numpy()))
                    psnrs.append(float(psnr(show_gs1, pred_gs1)))

                    milestone_root = save_root
                    store_image_path = milestone_root / "image"
                    store_flow_path = milestone_root / "flow"
                    store_pred_flow_path = milestone_root / "pred_flow"
                    store_true_flow_path = milestone_root / "true_flow"
                    store_light_path = milestone_root / "light"
                    store_wm_path = milestone_root / "wm"
                    store_output_path = milestone_root / "output"
                    img_name = f"{name}.png"
                    npy_name = f"{name}.npy"

                    for path in [
                        store_image_path,
                        store_flow_path,
                        store_pred_flow_path,
                        store_true_flow_path,
                        store_light_path,
                        store_wm_path,
                        store_output_path,
                    ]:
                        path.mkdir(exist_ok=True, parents=True)

                    utils.save_image(result_matrix, store_image_path / img_name)
                    utils.save_image(wm1, store_wm_path / img_name)
                    utils.save_image(light1, store_light_path / img_name)
                    utils.save_image(pred_gs1, store_output_path / img_name)

                    flow1 = flow_warp_show1[i].detach().cpu().numpy().transpose((1, 2, 0))
                    flow2 = flow_warp_show2[i].detach().cpu().numpy().transpose((1, 2, 0))
                    np.save(store_pred_flow_path / npy_name, flow1)
                    np.save(store_true_flow_path / npy_name, flow2)

                    flow1 = flow_to_image(flow1)
                    flow2 = flow_to_image(flow2)
                    result_flow = np.concatenate((flow1, flow2), axis=1)
                    cv2.imwrite(str(store_flow_path / img_name), result_flow)

            self.accelerator.log({"SSIM": float(np.mean(ssims))}, step=self.step)
            self.accelerator.log({"PSNR": np.mean(psnrs)}, step=self.step)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            data1 = None
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data1 = next(self.dl)
                    flow = data1[0].to(device)
                    rs = data1[1].to(device)
                    gs = data1[2].to(device)

                    flow = upsample2d_flow_as(flow, rs, mode="bilinear", if_rate=True)
                    norm_flow1 = norm_flow(flow).to(device)

                    class1 = torch.zeros(rs.shape[0]).to(device).to(torch.int64)
                    with self.accelerator.autocast():
                        loss = self.model(norm_flow1, classes=class1, condition=rs, gt=gs)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                    self.accelerator.log({"training_loss": loss}, step=self.step)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        self.sample(self.results_folder / f"sample-{milestone}", device)

                pbar.update(1)

        accelerator.print("training complete")
