"""
inference_utils.py
===================
NEW, opt-in inference-time utilities: test-time augmentation (TTA) and
post-hoc temperature scaling for probability calibration.

Neither touches training/checkpoint format -- both operate purely at
inference time, wrapping an already-trained, already-loaded model. Calling
the model directly as before (model(x, input_chans=...)) is completely
unaffected and gives identical results to before; these are additional,
explicitly-invoked entry points that a caller opts into.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def predict_with_tta(model, x, input_chans=None, n_views=5, noise_std=0.02, is_binary=True):
    """Average predictions over n_views noisy-augmented copies of x (the
    first view is always the original, unaugmented input) for a more
    robust probability estimate. Purely an inference-time wrapper -- does
    not change the model or its weights; calling model(x, ...) directly
    still gives exactly the single-pass result as before."""
    model.eval()
    views = [x]
    for _ in range(max(0, n_views - 1)):
        views.append(x + noise_std * x.std() * torch.randn_like(x))

    outs = [model(v, input_chans=input_chans) for v in views]
    stacked = torch.stack(outs, dim=0)
    if is_binary:
        probs = torch.sigmoid(stacked).mean(dim=0)
    else:
        probs = F.softmax(stacked, dim=-1).mean(dim=0)
    return probs


class TemperatureScaler(nn.Module):
    """Post-hoc probability calibration (Guo, Pleiss, Sun & Weinberger,
    "On Calibration of Modern Neural Networks", 2017): learns a single
    scalar temperature T by minimizing NLL of (logits / T) against true
    labels on a held-out calibration set. Dividing by a positive scalar
    does not change the argmax/ranking of the logits at all -- it only
    reshapes how confident the resulting probabilities look. Wraps a
    frozen, already-trained model; the base model's own forward()/
    state_dict/weights are completely untouched (self.model stays in eval
    mode with requires_grad unaffected -- only self.temperature is
    trainable)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, input_chans=None):
        with torch.no_grad():
            logits = self.model(x, input_chans=input_chans)
        return logits / self.temperature

    @torch.enable_grad()
    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, is_binary: bool = True,
           lr: float = 0.01, max_iter: int = 50):
        """val_logits: raw (pre-sigmoid/softmax) model outputs on a held-out
        calibration split, already computed with the frozen model. Returns
        the fitted temperature (float)."""
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Standard ECE metric (Guo et al. 2017) for a binary probability
    vector and 0/1 labels -- useful to check whether TemperatureScaler.fit
    actually improved calibration on a held-out set."""
    probs = probs.detach().flatten()
    labels = labels.detach().flatten().float()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = labels[in_bin].mean()
        bin_conf = probs[in_bin].mean()
        ece += (in_bin.float().mean()) * (bin_acc - bin_conf).abs()
    return ece.item()
