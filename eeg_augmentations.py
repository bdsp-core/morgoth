"""
eeg_augmentations.py
=====================
NEW, opt-in data augmentations for EEG classification finetuning:
SpecAugment-style masking, random channel dropout, and time warping.

Pure additions -- these are plain functions with no side effects on any
existing module. They only affect training when explicitly wired in via
new, default-off CLI flags in finetune_classification.py
(--specaugment / --channel_dropout / --time_warp, all default 0/False).
Not calling these functions (the default) leaves training data completely
unmodified, exactly as before.
"""
import torch
import torch.nn.functional as F


def spec_augment(x: torch.Tensor, time_mask_param: int = 2, freq_mask_param: int = 20,
                 n_time_masks: int = 1, n_freq_masks: int = 1) -> torch.Tensor:
    """x: [B, N_channels, A, T] patch-shaped EEG (A = number of patches,
    T = patch_size samples). Zeroes out contiguous ranges along the patch-
    index axis (A, analogous to SpecAugment's time masking) and the within-
    patch-sample axis (T, treated as the "frequency-like" axis here, since
    raw EEG patches have no spectrogram) -- same idea as SpecAugment (Park
    et al., "SpecAugment: A Simple Data Augmentation Method for Automatic
    Speech Recognition", 2019), applied to this patch-tensor layout instead
    of a mel-spectrogram."""
    x = x.clone()
    B, N, A, T = x.shape
    for _ in range(n_time_masks):
        span = min(time_mask_param, A)
        if span <= 0:
            continue
        t0 = torch.randint(0, A - span + 1, (1,)).item()
        t_len = torch.randint(0, span + 1, (1,)).item()
        x[:, :, t0:t0 + t_len, :] = 0
    for _ in range(n_freq_masks):
        span = min(freq_mask_param, T)
        if span <= 0:
            continue
        f0 = torch.randint(0, T - span + 1, (1,)).item()
        f_len = torch.randint(0, span + 1, (1,)).item()
        x[:, :, :, f0:f0 + f_len] = 0
    return x


def channel_dropout(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """x: [B, N_channels, A, T]. Independently zeroes out entire channels
    per sample with probability drop_prob -- a structured-dropout
    augmentation that trains the model to be robust to missing/noisy
    electrodes, a common real-world EEG failure mode."""
    if drop_prob <= 0:
        return x
    B, N, A, T = x.shape
    keep_mask = (torch.rand(B, N, 1, 1, device=x.device) >= drop_prob).to(x.dtype)
    return x * keep_mask


def time_warp(x: torch.Tensor, max_warp: float = 0.1) -> torch.Tensor:
    """x: [B, N_channels, A, T]. Randomly stretches or compresses the patch
    (time) axis by up to max_warp fraction via 1D linear interpolation,
    then center-crops/pads back to the original number of patches A, so
    the output shape always matches the input shape."""
    if max_warp <= 0:
        return x
    B, N, A, T = x.shape
    scale = 1.0 + (torch.rand(1).item() * 2 - 1) * max_warp
    new_A = max(1, int(round(A * scale)))

    flat = x.permute(0, 1, 3, 2).reshape(B * N, T, A)
    warped = F.interpolate(flat, size=new_A, mode='linear', align_corners=False)

    if new_A >= A:
        start = (new_A - A) // 2
        warped = warped[:, :, start:start + A]
    else:
        pad_total = A - new_A
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        warped = F.pad(warped, (pad_left, pad_right), mode='replicate')

    return warped.reshape(B, N, T, A).permute(0, 1, 3, 2)
