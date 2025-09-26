import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from pathlib import Path
import argparse
import torch
import mne
mne.set_log_level("ERROR")
import re
import torch.distributed as dist
from torch import inf
from torch import optim as optim
from tensorboardX import SummaryWriter
import bisect
import pickle
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
from timm.utils import get_state_dict
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import mat73
import h5py
import hdf5storage
import logging
logging.basicConfig(level=logging.CRITICAL)
from scipy.signal import butter, filtfilt, iirnotch
import sys
from typing import List
from collections import Counter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
list_path = List[Path]


standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', \
    'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
]



standard_1020_19 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']

eeg_channels1  = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

# T3=T7  T5=P7 T4=T8  T6=P8
eeg_channels2  = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T8', 'P8', 'O2']

sleep_channels1=['F3', 'C3',  'O1',  'F4', 'C4', 'O2']

sleep_channels2=['F3-M2', 'C3-M2',  'O1-M2',  'F4-M1', 'C4-M1', 'O2-M1']

class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha, gamma_base=2, reduction='mean'):
        """
        Dynamic Focal Loss
        Args:
            alpha (list or tensor): Class-wise weights.
            gamma_base (float): Base gamma value for dynamic adjustment.
            reduction (str): Reduction method: 'mean', 'sum', or 'none'.
        """
        super(DynamicFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma_base = gamma_base
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (tensor): Model outputs (logits), shape (N, C).
            targets (tensor): Ground truth labels, shape (N,).
        Returns:
            Tensor: Focal Loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        gamma = self.gamma_base * (1 - pt)
        focal_loss = self.alpha[targets] * (1 - pt) ** gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        alpha = torch.as_tensor(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MarginLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Multi-class Margin Loss
        Args:
            margin (float): Margin parameter.
            reduction (str): Reduction method: 'mean', 'sum', or 'none'.
        """
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (tensor): Model outputs (logits), shape (N, C).
            targets (tensor): Ground truth labels, shape (N,).
        Returns:
            Tensor: Margin Loss value.
        """
        true_logits = logits[torch.arange(len(targets)), targets].unsqueeze(1)  # Shape: (N, 1)

        margin_loss = torch.clamp(self.margin - (true_logits - logits), min=0)  # Shape: (N, C)
        margin_loss[torch.arange(len(targets)), targets] = 0  # 正确类别不参与损失计算

        if self.reduction == 'mean':
            return margin_loss.mean()
        elif self.reduction == 'sum':
            return margin_loss.sum()
        else:
            return margin_loss


class GHMC(nn.Module):
    def __init__(self, bins=10, momentum=0.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins)

    def forward(self, pred, target):
        # pred: [N, C], target: [N]
        N, C = pred.size()
        device = pred.device
        edges = self.edges.to(device)
        mmt = self.momentum

        # Compute the gradient magnitude
        loss = F.cross_entropy(pred, target, reduction='none')
        g = torch.abs(pred.detach().softmax(dim=1).view(-1) - F.one_hot(target, C).float().view(-1))

        # Compute the bin index for each gradient magnitude
        g_bin = torch.bucketize(g, edges)

        # Count the number of samples in each bin
        one_hot = F.one_hot(g_bin, self.bins).float()
        if mmt > 0:
            acc_sum = self.acc_sum.to(device)
            acc_sum = acc_sum * mmt + one_hot.sum(dim=0) * (1 - mmt)
            self.acc_sum = acc_sum.cpu()
        else:
            acc_sum = one_hot.sum(dim=0)

        # Compute the density of each bin
        num_valid = (acc_sum > 0).sum()
        acc_sum = torch.clamp(acc_sum, min=1)
        density = acc_sum / num_valid

        # Compute the GHM Loss
        ghm_loss = loss / density[g_bin]
        ghm_loss = ghm_loss.sum() / N

        return ghm_loss


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if len(iterable)!=0:
            average_time = total_time / len(iterable)
        else: average_time=total_time
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, average_time))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    # print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler =  torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
                 layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))

        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None,
               save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        print(f'[*] Saving model {epoch_name}')
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name), ]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        print('Using deepseek to save model...')
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)

    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu',weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'],strict=False)  # strict: bool=True, , strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1  # checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def load_from_task_model(args, model_without_ddp,):
    checkpoint = torch.load(args.task_model, map_location='cpu',weights_only=False)
    model_without_ddp.load_state_dict(checkpoint['model'],strict=False)  # strict: bool=True, , strict=False
    #print("Resume checkpoint %s" % args.resume)


def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size, start_percentage, end_percentage)
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list

def get_input_chans(ch_names):
    input_chans = [0]  # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name.upper()) + 1)
    return input_chans



def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (
                len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:  # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None,
                     **kwargs):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


def spikes_results_from_1s_to_10s(predictions,data_fs, original_result_step,original_step_in_point,new_result_step_second):
    if original_step_in_point:
        new_result_step = new_result_step_second*data_fs
        window_size = int(10 * data_fs / original_result_step)
    else:
        new_result_step=new_result_step_second
        window_size = int(10 / original_result_step)

    results_10s=[]
    for i in range(0,len(predictions)-1,new_result_step//original_result_step):

        results_1s=predictions[i:i+window_size]

        filtered_values = results_1s[results_1s >= 0.5]

        if len(filtered_values) > data_fs/4/original_result_step: # 0.25s
            avg_value = filtered_values.mean()
        else:
            avg_value = results_1s.mean()

        results_10s.append(avg_value)

    return results_10s


def exponential_moving_average(data, alpha=0.2):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def continuous_binary_probabilities(predictions, data_fs, result_step,smooth_method = 'ema'):
    if result_step > int(data_fs / 8):
        return predictions

    if len(predictions)*result_step < 2 * data_fs:
        return predictions

    window_size = int(data_fs/5/result_step) # window 0.2 second, 0.5 second in last version
    step = window_size

    if window_size>=len(predictions):
        return predictions

    smoothed_data = np.copy(predictions)

    if smooth_method == 'ema':
        smoothed_data=exponential_moving_average(smoothed_data,alpha=0.2)

    elif smooth_method == 'window_ema':

        for start in range(0, len(predictions)-window_size, step):
            end = min(len(predictions), start + window_size + 1)
            window = np.array(predictions[start:end])

            if np.sum(window < 0.5) > window_size: #* 3 / 4
                window = np.zeros_like(window)
            else:
                window = exponential_moving_average(window, alpha=0.1)
            smoothed_data[start:end] = window
    else:
        return smoothed_data

    return smoothed_data


def continuous_binary_probabilities2(predictions, data_fs, result_step):

    n = len(predictions)
    continuous_size = int(data_fs / result_step / 4)  # 1/4 秒

    keep_indices = [False] * n

    i = 0
    while i < n:
        if predictions[i] < 0.5:
            i += 1
            continue
        start = i
        while i < n and predictions[i] >= 0.5:
            i += 1
        end = i
        if end - start > continuous_size:
            keep_start = max(start - continuous_size*2, 0)
            keep_end = min(end + continuous_size*2, n)

            for j in range(keep_start, keep_end):
                keep_indices[j] = True

    for i in range(n):
        if not keep_indices[i]:
            predictions[i] = max(predictions[i] - 0.5, 0)

    return predictions

def continuous_multiclass_probabilities(prediction_matrix, data_fs,result_step, use_smooth=True):
    if not use_smooth:
        return prediction_matrix

    if result_step > int(data_fs * 10) or prediction_matrix.shape[0] <3:
        return prediction_matrix


    predictions = np.argmax(prediction_matrix, axis=1)

    for i in range(1, len(predictions) - 1):
        if predictions[i] != predictions[i - 1] and predictions[i-1] == predictions[i + 1]:
            prediction_matrix[i] = (prediction_matrix[i - 1] + prediction_matrix[i + 1]) / 2

    return prediction_matrix


def prepare_classification_dataset(root,original_format='pkl', Bipolar=False, addBipolar=False, only_evaluate=False, sub_dir=False, sample_length=False,):
    if only_evaluate:
        print(f"[*] For {sub_dir}.......")
        result_file = os.path.join(sub_dir, 'pred.csv')
        evaluation_files = sorted([f for f in os.listdir(sub_dir) if f.endswith(original_format)])

        if len(evaluation_files)==0:
            print(f'{sub_dir} dose not have {original_format} data')
            return False,False

        df = pd.DataFrame({
            'data': evaluation_files
        })
        df.to_csv(result_file, index=False)

        if 'TUEV' in result_file:
            evaluation_dataset=TUEVLoader(sub_dir, evaluation_files)

        else:
            evaluation_dataset = MGBClassLoader(sub_dir, evaluation_files, original_format=original_format, Bipolar=Bipolar, addBipolar=addBipolar,sample_length=sample_length)
        print(f'[*] Evaluation sizes: {len(evaluation_dataset)}')
        return result_file, evaluation_dataset

    else:
        train_files_path = glob.glob(os.path.join(root, "train", "*.pkl"))
        val_files_path = glob.glob(os.path.join(root, "val", "*.pkl"))
        test_files_path = glob.glob(os.path.join(root, "test", "*.pkl"))

        train_files = [os.path.basename(file) for file in train_files_path]
        val_files = [os.path.basename(file) for file in val_files_path]
        test_files = [os.path.basename(file) for file in test_files_path]

        # prepare training and test data loader
        train_dataset = MGBClassLoader(os.path.join(root, "train"), train_files, Bipolar=Bipolar, addBipolar=addBipolar, sample_length=sample_length)
        val_dataset = MGBClassLoader(os.path.join(root, "val"), val_files, Bipolar=Bipolar, addBipolar=addBipolar,
                                     sample_length=sample_length)
        test_dataset = MGBClassLoader(os.path.join(root, "test"), test_files, Bipolar=Bipolar, addBipolar=addBipolar,
                                      sample_length=sample_length)

        print(f'Train, Val, Test sizes: {len(train_files)}, {len(val_files)}, {len(test_files)}')
        return train_dataset, test_dataset, val_dataset



class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""

    def __init__(self, file_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx)  # the start index of the subject's sample in the dataset
            subject_len = self.__file[subject]['eeg'].shape[1]
            # total number of samples
            total_sample_num = (subject_len - self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx + self.__window_size]

    def free(self) -> None:
        if self.__file:
            self.__file.close()
            self.__file = None

    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""

    def __init__(self, file_paths: list_path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [
            SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage,
                               self.__end_percentage) for file_path in self.__file_paths]

        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]

    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()

    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y



class MGBClassLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, original_format='pkl', Bipolar=False, addBipolar=False, sample_length=False):
        self.root = root
        self.files = files
        # self.default_rate = 200
        # self.sampling_rate = sampling_rate # 不需要了，默认在evaluation的时候segment除了normalization才做，都是处理好了
        self.data_format = original_format
        self.Bipolar = Bipolar
        self.addBipolar = addBipolar
        self.sample_length = sample_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path_signal=os.path.join(self.root, self.files[index])
        if self.data_format  == 'mat':
            try:
                sample = mat73.loadmat(path_signal)
                X = sample['data']
                # X = X[:19, :]
                # X = EEG_avg(X)
                # X = EEG_clip(X)
                X = EEG_normalize(X)
                try:
                    Y = sample['y'].item()
                except KeyError:
                    Y = 0

            except TypeError:
                try:
                    sample = scipy.io.loadmat(path_signal)
                    X = sample['data']
            ############ data that was only filtered but not otherwise processed (when running Morgoth on test set)############
                    # X= X[:19,:]
                    # # print(X.shape)
                    # X = EEG_avg(X)
                    # X = EEG_clip(X)
            ############  data that was only filtered but not otherwise processed (when running Morgoth on test set)############
                    X = EEG_normalize(X)
                    try:
                        Y = sample['y'].item()
                    except KeyError:
                        Y = 0
                except Exception as e:
                    raise ValueError(f'Failed to load {path_signal}. Mat type error : {e}')
        else:
            sample = pickle.load(open(path_signal, "rb"))
            X = sample["X"]
            # X = EEG_avg(X)
            # X = EEG_clip(X)  # For .pkl data, avg and clip have already been handled in data_provider
            X = EEG_normalize(X)
            Y = sample["y"]
            if Y==-1:
                Y = 0

        if self.sample_length:
            if self.sample_length != 15:
                middle_start = (15 / 2 - self.sample_length / 2) * 200
                middle_end = (15 / 2 + self.sample_length / 2) * 200
                X = X[:, int(middle_start):int(middle_end)]
                X = X[:, :int(self.sample_length) * 200]

        if self.Bipolar:
            X, _ = bipolar(X)
            X = X[:, :2800]  # 200*int(256/18)=1200  about 14 seconds
        elif self.addBipolar:
            X_bi, _ = bipolar(X)
            X = np.vstack((X, X_bi))
            X = X[:, :1200]  # 200*int(256/37)=1200  about 6 seconds
        else:
            X = X[:, :2600]  # 200*int(256/19)=2600  about 13 seconds

        X = torch.FloatTensor(X)
        return X, Y




class ContinuousToSnippetDataset(torch.utils.data.Dataset):
    # Dataset that takes a continous signal and returns snippets of a given length
    # input shape: (n_channels,n_timepoints), output shape: (n_snippets,n_channels,ts)
    def __init__(self,type,
                 path_signal,
                 given_fs=0,
                 has_avg=False,
                 has_formatted_channel=False,
                 allow_missing_channels=False,
                 montage = None,
                 transform = None,
                 step = 2000,
                 step_in_point = True,
                 polarity=1,
                 leave_one_hemisphere_out=False,
                 channel_symmetric_flip=False,
                 max_length_hour=None):

        self.type = type
        self.fs = 200
        self.original_avg=False
        self.missing_channels=None
        self.mono_channels=None
        self.leave_one_hemisphere_out=leave_one_hemisphere_out
        self.channel_symmetric_flip=channel_symmetric_flip

        file_extension = os.path.splitext(path_signal)[1].lower()
        if file_extension == '.mat':
            try:
                raw= mat73.loadmat(path_signal)
                signal=raw['data']
                self_fs=get_frequency_from_mat(raw_mat=raw)

            except TypeError:
                try:
                    raw=scipy.io.loadmat(path_signal)
                    signal = raw['data']
                    self_fs = get_frequency_from_mat(raw_mat=raw)

                except Exception as e:
                    try:
                        raw = hdf5storage.loadmat(path_signal)
                        signal = raw['data']
                        self_fs = get_frequency_from_mat(raw)
                    except Exception as e:
                        raise ValueError(f'Failed to load {path_signal}. Mat type error : {e}')

            if has_formatted_channel:
                if self.type == 'SLEEPPSG':
                    if signal.shape[0]==19:
                        signal = signal[[1, 2, 7, 12, 13, 18], :]
                    else:
                        signal =signal[:6,:]
                    original_avg = False
                else:
                    signal=signal[:19,:]
                    original_avg=False

            else:
                # order channels 如果原数据不是排好序的，要求mat里提供channel name
                channels, original_avg = get_channel_names_from_mat(raw)

                data_dic = dict(zip(channels, signal))

                if self.type == 'SLEEPPSG':
                    if set(channels).issuperset(set(sleep_channels1)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=sleep_channels1,
                                                     default_value=np.array([0] * signal.shape[1]))
                    elif set(channels).issuperset(set(sleep_channels2)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=sleep_channels2,
                                                     default_value=np.array([0] * signal.shape[1]))
                    else:
                        raise ValueError(
                            "EDF file does not contain all channels from either sleep_channels1 or sleep_channels2.")
                else:
                    if set(channels).issuperset(set(eeg_channels1)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=eeg_channels1,
                                                     default_value=np.array([0] * signal.shape[1]))
                    elif set(channels).issuperset(set(eeg_channels2)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=eeg_channels2,
                                                     default_value=np.array([0] * signal.shape[1]))
                    else:
                        if allow_missing_channels:
                            missing_channels1 = set(eeg_channels1) - set(channels)
                            missing_channels2 = set(eeg_channels2) - set(channels)

                            if len(missing_channels1) <= len(missing_channels2):
                                self.mono_channels = eeg_channels1
                                self.missing_channels = [ch for ch in eeg_channels1 if ch not in channels]
                                selected_channels = [ch for ch in eeg_channels1 if ch in channels]
                            else:
                                self.mono_channels = eeg_channels2
                                self.missing_channels = [ch for ch in eeg_channels2 if ch not in channels]
                                selected_channels = [ch for ch in eeg_channels2 if ch in channels]

                            data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=selected_channels,
                                                         default_value=np.array([0] * signal.shape[1]))

                        else:
                            raise ValueError("Mat file does not contain all EEG channels")

                signal = np.array(list(data_dic.values()))

            if max_length_hour is not None and signal.shape[1] > int(max_length_hour * 3600 * self_fs):
                signal = signal[:, :int(max_length_hour * 3600 * self_fs)]

            if polarity == -1:
                signal = signal * -1

            if original_avg or has_avg:
                self.original_avg = True
            else:
                self.original_avg = False

        elif file_extension == '.edf':
            try:
                raw = mne.io.read_raw_edf(path_signal, preload=True)
            except Exception as e:
                raise ValueError(f"Edf file {path_signal} corrupted: {e}")

            # may have EEG in name initially
            raw.rename_channels(
                {name: name.replace('EEG', '').replace('eeg', '').replace('POL', '').replace('pol', '').strip() for name in raw.info['ch_names']})

            new_channel_names = {ch_name: ch_name.upper() for ch_name in raw.ch_names}
            raw.rename_channels(new_channel_names)
            channels = raw.ch_names

            if '-AVG' in channels[0]:
                self.original_avg = True
                # channels = [ch.replace('-AVG', '') for ch in channels]
            elif has_avg:
                self.original_avg = True
            else:
                self.original_avg = False

            # Remove the reference name in the channel names
            # new_channel_names={ch_name: ch_name.split('-')[0] for ch_name in channels}
            # raw.rename_channels(new_channel_names)

            # Remove the reference name in the channel names 
            # Create a mapping of new channel names

            # new_channel_names = {ch_name: ch_name.split('-')[0] for ch_name in channels}
            # new_channel_names = {ch_name: ch_name.split('(')[0] for ch_name in new_channel_names}

            new_channel_names = {
                ch_name: re.sub(r"\(.*?\)", "", ch_name).split('-')[0].strip()
                for ch_name in channels
            }

            # Count occurrences of each new channel name
            counter = Counter(new_channel_names.values())
            # Create final mapping: only rename unique channels
            final_channel_names = {}
            for old_name, new_name in new_channel_names.items():
                if counter[new_name] == 1:  # Only rename if unique
                    final_channel_names[old_name] = new_name
                else:
                    final_channel_names[old_name] = old_name  # Keep original name if duplicate
            # Apply the updated names
            raw.rename_channels(final_channel_names)


            channels=raw.ch_names

            if self.type=='SLEEPPSG':
                if set(channels).issuperset(set(sleep_channels1)):
                    selected_channels = sleep_channels1
                elif set(channels).issuperset(set(sleep_channels2)):
                    selected_channels = sleep_channels2
                else:
                    missing = set(sleep_channels1) - set(channels)
                    if not missing:
                        missing = set(sleep_channels2) - set(channels)
                    raise ValueError(
                        f"{path_signal} EDF file does not contain all channels from either sleep_channels1 or sleep_channels. Missing {missing}")

            else:
                # Missing channels are allowed and will be padded with zeros.
                if allow_missing_channels:
                    missing_channels1 = set(eeg_channels1) - set(channels)
                    missing_channels2 = set(eeg_channels2) - set(channels)
                    if len(missing_channels1) <= len(missing_channels2):
                        self.mono_channels=eeg_channels1
                        self.missing_channels = [ch for ch in eeg_channels1 if ch not in channels]
                        selected_channels = [ch for ch in eeg_channels1 if ch in channels]
                    else:
                        self.mono_channels = eeg_channels2
                        self.missing_channels = [ch for ch in eeg_channels2 if ch not in channels]
                        selected_channels = [ch for ch in eeg_channels2 if ch in channels]
                else:
                    if set(channels).issuperset(set(eeg_channels1)):
                        selected_channels = eeg_channels1
                    elif set(channels).issuperset(set(eeg_channels2)):
                        selected_channels = eeg_channels2
                    else:
                        missing = set(eeg_channels1) - set(channels)
                        if not missing:
                            missing = set(eeg_channels2) - set(channels)
                        raise ValueError(f"{path_signal} EDF file does not contain all channels from either eeg_channels1 or eeg_channels2. Missing {missing}")

            raw_selected = raw.copy().pick(selected_channels)
            if max_length_hour is not None and raw_selected.times[-1] >  max_length_hour * 3600:
                raw_selected.crop(tmin=0, tmax=int(max_length_hour * 3600))
            signal=raw_selected.get_data(units='uV')

            ############### for HEP data, should *10 before input the model ################
            # signal=signal*10
            # print('hep\'s signal * 10')
            ############### for HEP data, should *10 before input the model ################
            if polarity == -1:
                signal = signal * -1

            self_fs = int(raw.info['sfreq'])

        elif file_extension == '.pkl':
            with open(path_signal, 'rb') as f:
                eeg_data = pickle.load(f)
            signal=eeg_data['X'][:19,:]
            self_fs = given_fs
            self.original_avg=False

        else:
            raise ValueError("Should be mat or edf or pkl.")

        # If the data file includes fs, use it if it differs from the provided parameter
        if self_fs == 0:
            if given_fs != 0:
                self_fs = given_fs
            else:
                raise ValueError(f'{path_signal} has no sampling rate in data file or input parameter')

        elif given_fs != 0 and self_fs != given_fs:
            print('Input sampling rate dose not match recorded sampling rate, use recorded')

        self.self_fs = self_fs

        ######## Only test on center n s(compare with non-continuous baselines) ##################
        # signal=same_segment_with_kaggle(signal=signal,fs=self.self_fs,seq_length=30)
        ######## Only test on center n s(compare with non-continuous baselines) ##################

        # index of flat value
        def is_constant(tensor):
            max_values = torch.max(tensor, dim=1).values
            min_values = torch.min(tensor, dim=1).values
            diff = max_values - min_values
            return torch.all(diff < 1).item()

        def expand_zeros(lst):
            lst = np.array(lst)
            zero_mask = (lst == 0)

            left_shift = np.roll(zero_mask, shift=1)
            right_shift = np.roll(zero_mask, shift=-1)

            left_shift[0] = False
            right_shift[-1] = False

            lst[(left_shift | right_shift) & (lst == 1)] = 0
            return lst.tolist()

        window_size=self._get_window_size()

        if step_in_point:
            original_step = step
            new_step = max(int(original_step / self.self_fs * self.fs), 1)  
        else:
            original_step = int(step * self.self_fs)
            new_step = int(step * self.fs)

        original_window_size=window_size * self.self_fs # generate snippets of shape (n_snippets,n_channels,ts)
        new_window_size = window_size * self.fs

        original_signal = torch.FloatTensor(signal.astype(np.float32))
        original_snippets = original_signal.unfold(dimension=1, size=original_window_size, step=original_step).permute(1, 0, 2)

        num_snippets, num_channels, time_steps = original_snippets.shape
        self.valid_data_index = [0]*num_snippets

        cut_min = 20 if "SPIKE" in self.type else 2

        for snippet_idx in range(num_snippets):
            snippet_data = original_snippets[snippet_idx, :, :]
            is_valid = 1

            if torch.any(torch.all(torch.isnan(snippet_data), dim=0)).item():
                is_valid = 0

            if not self.original_avg and (torch.all(torch.abs(snippet_data) < cut_min) or torch.all(torch.abs(snippet_data) >3000)):
                is_valid = 0

            snippet_data -= torch.mean(snippet_data, dim=0, keepdim=True)
            if is_constant(snippet_data):
                is_valid=0

            self.valid_data_index[snippet_idx]=is_valid

        self.valid_data_index=expand_zeros(self.valid_data_index)
        if all(x == 0 for x in self.valid_data_index):
            raise SnippetsError(result_segment_shapes=int(num_snippets) ,message="EEG has no valid snippets")

        def find_continuous_valid_indices(arr):
            arr = np.array(arr)
            diff = np.diff(arr, prepend=0, append=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1
            return list(zip(starts, ends))

        self.valid_start_end_indices = find_continuous_valid_indices(self.valid_data_index)

        self.result_segment_shape=[]
        all_snippets = []
        for start_idx, end_idx in self.valid_start_end_indices:
            start_point= original_step*start_idx
            end_point= original_step*end_idx+original_window_size
            valid_signal=signal[:,start_point:end_point]

            if "SPIKE" in self.type and self.self_fs>128:
                valid_signal = resample_signal(valid_signal, original_rate=self.self_fs,target_rate=128)
                valid_signal = EEG_bandfilter(valid_signal, fs=128)
                valid_signal = EEG_notchfilter(valid_signal, fs=128)
                self_fs=128

            valid_signal = resample_signal(valid_signal, original_rate=self_fs, target_rate=self.fs)
            valid_signal = EEG_bandfilter(valid_signal, fs=self.fs)
            valid_signal = EEG_notchfilter(valid_signal, fs=self.fs)

            # move signal to torch
            valid_signal = torch.FloatTensor(valid_signal.astype(np.float32))
            # generate snippets of shape (n_snippets,n_channels,ts)
            snippet=valid_signal.unfold(dimension=1, size=new_window_size, step=new_step).permute(1, 0, 2)
            all_snippets.append(snippet)

            self.result_segment_shape.append(snippet.shape[0])

        self.snippets = torch.cat(all_snippets, dim=0)

        # set montage
        self.montage = montage
        # set transform
        self.transform = transform


    def __len__(self):
        # get item zero of self. snippets, which has shape (n_snippets,n_channels,ts)
        return self.snippets.shape[0]

    def _preprocess(self, signal):
        '''preprocess signal and apply montage, transform and normalization'''

        # apply montage: avg
        if self.montage is not None:

            signal = self.montage(signal,mono_channels=self.mono_channels,missing_channels=self.missing_channels)

        # apply transformations: clip and scaling (normalize)
        if self.transform is not None:
            signal = self.transform(signal)

        if self.leave_one_hemisphere_out is not False:
            signal=leave_one_hemisphere_out_func(data=signal,side=self.leave_one_hemisphere_out)
        if self.channel_symmetric_flip is not False:
            signal = channel_symmetric_flip_func(data=signal,side=self.channel_symmetric_flip)

        # transfer to torch
        if isinstance(signal, np.ndarray):
            signal = torch.FloatTensor(signal.copy())

        return signal

    def __getitem__(self, idx):
        # get the snippet
        # print(self.snippets)
        signal = self.snippets[idx, :, :]
        # preprocess signal
        signal = self._preprocess(signal)

        # return signal and dummy label, the latter to prevent lightning dataloader from complaining
        return signal,0

    def _get_window_size(self):
        if self.type=='SPIKES':
            return 1
        else: return 10

    def get_valid_indices(self):
        return self.valid_data_index, self.valid_start_end_indices, self.result_segment_shape

    def get_original_fs(self):
        return self.self_fs

def get_n_classes(dataset):
    n_class_map={
        'NORMAL': 1,
        'BS':1,
        'SPIKES':1,
        'FOC_GEN_SPIKES':3,
        'SLOWING':3,
        'IIIC':6,
        'MGBSLEEP3stages': 3,
        'SLEEPPSG':5,
    }
    return n_class_map[dataset]


class common_average_montage():
    def __init__(self):
        self.mono_channels = eeg_channels1
        # self.channel_average = ['FP1-avg', 'F3-avg', 'C3-avg', 'P3-avg', 'F7-avg', 'T3-avg', 'T5-avg', 'O1-avg', 'FZ-avg', 'CZ-avg', 'PZ-avg', 'FP2-avg', 'F4-avg', 'C4-avg', 'P4-avg', 'F8-avg', 'T4-avg', 'T6-avg', 'O2-avg']  # 19
        #self.average_ids = [self.mono_channels.index(ch.split('-')[0]) for ch in self.channel_average]

    def __call__(self, signal, mono_channels=None, missing_channels=None):
        # Common Average Montage
        # signal = signal[self.average_ids]
        column_means = torch.mean(signal, dim=0, keepdim=True)
        data_centered = signal - column_means

        if missing_channels is not None:
            self.mono_channels = mono_channels
            valid_channels = [ch for ch in self.mono_channels if ch not in missing_channels]
            valid_indices = [self.mono_channels.index(ch) for ch in valid_channels]

            num_channels = len(self.mono_channels)
            num_samples = data_centered.size(1)
            full_signal = torch.zeros((num_channels, num_samples))

            for idx, valid_idx in enumerate(valid_indices):
                full_signal[valid_idx, :] = data_centered[idx, :]

            data_centered=full_signal
        return data_centered

    def get_channel_names(self):
        return self.mono_channels


class sleep_common_average_montage():
    def __init__(self):
        self.mono_channels =['F3', 'C3',  'O1',  'F4', 'C4', 'O2']

        # self.channel_average =  ['F3-avg', 'C3-avg', 'O1-avg', 'F4-avg', 'C4-avg', 'O2-avg']
        # self.average_ids = [self.mono_channels.index(ch.split('-')[0]) for ch in self.channel_average]

    def __call__(self, signal,mono_channels=None,missing_channels=None):
        # Common Average Montage
        # signal=signal[self.average_ids]
        column_means = torch.mean(signal, dim=0, keepdim=True)
        data_centered = signal - column_means

        return data_centered

    def get_channel_names(self):
        return self.mono_channels

class bipolar_montage():
    def __init__(self):
        self.mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        self.bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ'] # 18


        self.bipolar_ids = np.array(
            [[self.mono_channels.index(bc.split('-')[0]), self.mono_channels.index(bc.split('-')[1])] for bc in self.bipolar_channels])

    def __call__(self, signal):
        # Bipolar Montage
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]
        return bipolar_signal

    def get_channel_names(self):
        return self.bipolar_channels


def bipolar(data_monopolar):
    # Initialize the bipolar data array
    data_bipolar = np.zeros((18, data_monopolar.shape[1]))
    # Group LL
    data_bipolar[0, :] = data_monopolar[0, :] - data_monopolar[4, :]  # Fp1-F7
    data_bipolar[1, :] = data_monopolar[4, :] - data_monopolar[5, :]  # F7-T3
    data_bipolar[2, :] = data_monopolar[5, :] - data_monopolar[6, :]  # T3-T5
    data_bipolar[3, :] = data_monopolar[6, :] - data_monopolar[7, :]  # T5-O1

    # Group RL
    data_bipolar[4, :] = data_monopolar[11, :] - data_monopolar[15, :]  # Fp2-F8
    data_bipolar[5, :] = data_monopolar[15, :] - data_monopolar[16, :]  # F8-T4
    data_bipolar[6, :] = data_monopolar[16, :] - data_monopolar[17, :]  # T4-T6
    data_bipolar[7, :] = data_monopolar[17, :] - data_monopolar[18, :]  # T6-O2

    # Group LP
    data_bipolar[8, :] = data_monopolar[0, :] - data_monopolar[1, :]  # Fp1-F3
    data_bipolar[9, :] = data_monopolar[1, :] - data_monopolar[2, :]  # F3-C3
    data_bipolar[10, :] = data_monopolar[2, :] - data_monopolar[3, :]  # C3-P3
    data_bipolar[11, :] = data_monopolar[3, :] - data_monopolar[7, :]  # P3-O1

    # Group RP
    data_bipolar[12, :] = data_monopolar[11, :] - data_monopolar[12, :]  # Fp2-F4
    data_bipolar[13, :] = data_monopolar[12, :] - data_monopolar[13, :]  # F4-C4
    data_bipolar[14, :] = data_monopolar[13, :] - data_monopolar[14, :]  # C4-P4
    data_bipolar[15, :] = data_monopolar[14, :] - data_monopolar[18, :]  # P4-O2

    # Group midline
    data_bipolar[16, :] = data_monopolar[8, :] - data_monopolar[9, :]  # Fz-Cz
    data_bipolar[17, :] = data_monopolar[9, :] - data_monopolar[10, :]  # Cz-Pz

    # 10-20 system
    channel_names = ["Fp1-F7","F7-T3","T3-T5","T5-O1","Fp2-F8","F8-T4","T4-T6","T6-O2","Fp1-F3","F3-C3","C3-P3","P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2","Fz-Cz","Cz-Pz"]
    # MCN system, coresponding to the above

    # channel_names=["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1","FP2-F4", "F4-C4", "C4-P4", "P4-O2"]

    return data_bipolar,channel_names

class combine_montage():
    def __init__(self):
        self.mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4',
                         'F8', 'T4', 'T6', 'O2']
        self.bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']  # 18
        self.channel_average = ['FP1-avg', 'F3-avg', 'C3-avg', 'P3-avg', 'F7-avg', 'T3-avg', 'T5-avg', 'O1-avg', 'FZ-avg', 'CZ-avg', 'PZ-avg', 'FP2-avg', 'F4-avg', 'C4-avg', 'P4-avg', 'F8-avg', 'T4-avg', 'T6-avg',
                           'O2-avg']  # 19

        self.bipolar_ids = np.array(
            [[self.mono_channels.index(bc.split('-')[0]), self.mono_channels.index(bc.split('-')[1])] for bc in self.bipolar_channels])

        self.average_ids = [self.mono_channels.index(ch.split('-')[0]) for ch in self.channel_average]


    def __call__(self, signal):
        common_average_signal =  signal[self.average_ids] - torch.mean(signal[self.average_ids], dim=0, keepdim=True)
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]

        combined_signal = np.vstack([common_average_signal,bipolar_signal])

        return combined_signal

    def get_channel_names(self):
        return  self.mono_channels+self.bipolar_channels


class single_channel_average_montage():
    def __init__(self,channel_idx):
        self.channel_idx = [channel_idx]
        self.mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

    def __call__(self, signal):
        # Common Average Montage
        signal=signal[self.channel_idx]
        column_means = torch.mean(signal, dim=0, keepdim=True)
        data_centered = signal - column_means

        return data_centered

    def get_channel_names(self):
        return [self.mono_channels[self.channel_idx[0]]]


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class Clipping:
    def __init__(self,clip_at = 500):
        self.clip_at = clip_at #microV

    def __call__(self,signal):
        out_data = np.clip(signal, -self.clip_at, self.clip_at)
        return out_data

class Scaling:
    def __init__(self, scaling_at=100):
        self.scaling_at = scaling_at  # microV

    def __call__(self, signal):
        normalized_eeg_data = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            channel_data = signal[i, :].reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(-self.scaling_at, self.scaling_at))
            normalized_channel_data = scaler.fit_transform(channel_data)
            normalized_eeg_data[i, :] = normalized_channel_data.flatten()
        return normalized_eeg_data


class Normalize: # not use
    def __init__(self, q=0.95):
        self.q = q  # microV

    def __call__(self, signal):
        normalized_eeg_data = signal / (np.quantile(np.abs(signal), q=self.q, method="linear", axis=-1, keepdims=True) + 1e-8)
        return normalized_eeg_data


def EEG_bandfilter(data, fs, order=4, low=0.5, high=70):
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    if high>1:
        b, a = butter(order, low, btype='high') # only highpass
    else:
        b, a = butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, data[i, :])
    return filtered_data


def EEG_notchfilter(data, fs=200, notch_width=1.0):
    Q_50 = 50 / notch_width
    b_50, a_50 = iirnotch(50, Q_50, fs)

    Q_60 = 60 / notch_width
    b_60, a_60 = iirnotch(60, Q_60, fs)

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b_50, a_50, data[i, :])
        filtered_data[i, :] = filtfilt(b_60, a_60, filtered_data[i, :])
    return filtered_data


def replace_nan_with_channel_mean(data):
    for channel in range(data.shape[0]):
        channel_data = data[channel, :]
        channel_mean = np.nanmean(channel_data)
        channel_data[np.isnan(channel_data)] = channel_mean
        data[channel, :] = channel_data
    return data

def replace_nan_with_zero(data):
    data[np.isnan(data)] = 0
    return data

def interpolate_nan(signal):
    nans = np.isnan(signal)
    if np.any(nans):
        x = np.arange(len(signal))
        signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    return signal

def EEG_avg(eeg_data):
    avg = np.mean(eeg_data, axis=0)
    out_data = eeg_data - avg[np.newaxis, :]
    return out_data

def EEG_clip(eeg_data):
    out_data = np.clip(eeg_data, -500, 500)
    return out_data

def EEG_normalize(eeg_data):
    out_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        channel_data = eeg_data[i, :].reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-100, 100))
        normalized_channel_data = scaler.fit_transform(channel_data)
        out_data[i, :] = normalized_channel_data.flatten()
    return out_data

def resample_signal(signal, original_rate, target_rate, n_jobs=5):

    if original_rate == target_rate:
        return signal
    # num_samples = int(signal.shape[1] * (target_rate / original_rate))
    # resampled_signal = np.zeros((signal.shape[0], num_samples))
    # for i in range(signal.shape[0]):
    #     resampled_signal[i, :] = resample(signal[i, :], num_samples)

    resampled_signal = mne.filter.resample(signal, down=original_rate, up=target_rate, n_jobs=n_jobs)
    return resampled_signal


def get_frequency_from_mat(raw_mat):
    try:
        fs_value = raw_mat['Fs']
    except KeyError:
        try:
            fs_value = raw_mat['fs']
        except KeyError:
            try:
                fs_value = raw_mat['sampling_rate']
            except KeyError:
                    return 0

    if isinstance(fs_value, np.ndarray):
        if fs_value.shape == (1, 1, 1):
            fs_value = fs_value[0, 0, 0]
        elif fs_value.shape == (1, 1):
            fs_value = fs_value[0, 0]
        elif fs_value.shape == (1,):
            fs_value = fs_value[0]
        elif fs_value.shape == ():
            fs_value=fs_value.item()
        else:
            print('Unexpected array shape for fs value in mat')
            return 0

    if isinstance(fs_value, np.ndarray):
        fs_value = fs_value.item()

    return int(fs_value)


def get_channel_names_from_mat(raw_mat):
    """
    Extract channel names from the channels array and remove leading/trailing spaces.
    :param channels: Channels array
    :return: List of channel names
    """
    try:
        channels = raw_mat['channels']
    except KeyError:
        try:
            channels = raw_mat['channel_locations']
        except KeyError:
            raise ValueError(f'No channel names found in mat')

    channel_names = []
    # Iterate through the channels array
    for channel in channels:
        # Handle different cases
        if isinstance(channel, np.ndarray):
            # If channel is a nested array
            if channel.size == 1:
                channel_name = channel.item()
            else:
                channel_name = channel[0]

            # Further unwrap if necessary
            if isinstance(channel_name, np.ndarray):
                if channel_name.size == 1:
                    channel_name = channel_name.item()
                else:
                    channel_name = channel_name[0]

                # Further unwrap if necessary
                if isinstance(channel_name, np.ndarray):
                    if channel_name.size == 1:
                        channel_name = channel_name.item()
                    else:
                        channel_name = channel_name[0]
        elif isinstance(channel, list):
            # If channel is a list
            channel_name = channel[0]
        else:
            # If channel is a single element
            channel_name = channel

        # Ensure the channel name is a string
        if isinstance(channel_name, np.ndarray):
            channel_name = channel_name.item()
        channel_names.append(channel_name.strip())

    channel_names=[ch.upper() for ch in channel_names]
    if '-AVG' in channel_names[0]:
        hasAVG=True
        # channel_names=[ch.replace('-AVG', '') for ch in channel_names]
    else:
        hasAVG=False

    channel_names = [ch.split('-')[0] for ch in channel_names]

    return channel_names, hasAVG


def sort_dict_by_keys(input_dict, key_order, default_value=None, remaining_keys=False):
    """
    Sorts the keys of a dictionary according to a specified order, where the keys in the order appear in the strings of the dictionary keys.

    :param input_dict: Input dictionary
    :param key_order: List of keys specifying the order
    :param default_value: Default value for keys in `key_order` that are not found in `input_dict`
    :return: Sorted dictionary
    """
    sorted_dict = {}

    for key in key_order:
        # Find matching keys
        matched_keys = [k for k in input_dict.keys() if re.search(key.upper(), k.upper())]

        if matched_keys:
            # If there are matching keys, select the first one
            matched_key = matched_keys[0]
            sorted_dict[matched_key] = input_dict[matched_key]
        else:
            # If there are no matching keys, use the default value
            sorted_dict[key] = default_value

    # Add remaining keys
    if remaining_keys:
        for k in input_dict.keys():
            if k not in sorted_dict:
                sorted_dict[k] = input_dict[k]

    return sorted_dict



def remove_nan_columns(data):
    nan_columns = np.all(np.isnan(data), axis=0)

    n_removed_front_points = 0
    for i in range(data.shape[1]):
        if nan_columns[i]:
            n_removed_front_points += 1
        else:
            break

    n_removed_back_points = 0
    for i in range(data.shape[1] - 1, -1, -1):
        if nan_columns[i]:
            n_removed_back_points += 1
        else:
            break

    cleaned_data = data[:, ~nan_columns]

    return cleaned_data, n_removed_front_points, n_removed_back_points


def find_valid_segment(data, n, threshold=5000):

    num_channels, num_columns = data.shape

    if n < 0 or n >= num_columns:
        print(f"index {n} not in [0, {num_columns - 1}]")
        return np.array([]), -1

    start = n
    end = n

    while start > 0:
        if np.all(np.abs(data[:, start - 1]) < threshold):
            start -= 1
        else:
            break

    while end < num_columns - 1:
        if np.all(np.abs(data[:, end + 1]) < threshold):
            end += 1
        else:
            break

    new_data = data[:, start:end + 1]

    new_n = n - start

    return new_data, new_n



def leave_one_hemisphere_out_func(data, side='right'):
    if side == 'right' or side == 'r':
        data[-8:, :] = 0
    elif  side == 'left' or side == 'l':
        data[:8, :] = 0
    elif side == 'middle' or side == 'm':
        data[8:11, :] = 0
    else:
        raise ValueError(f'Hemisphere side should be right or left or middle')
    return data

def channel_symmetric_flip_func(data,side='right'):
    if side == 'right' or side == 'r':
        data[:8, :] =  data[-8:, :]
    elif side == 'left' or side == 'l':
        data[-8:, :] = data[:8, :]
    else:
        raise ValueError(f'Hemisphere side should be right or left')
    return data


def resize_array_along_axis0(arr, d, target_length):

    original_length = arr.shape[0]

    if target_length == original_length:
        return arr

    indices = np.round(np.linspace(0, original_length - 1, target_length)).astype(int)

    if d==1:
        return arr[indices]
    else:
        # return arr[indices:,]
        return arr[indices,: ]

def split_nd_to_plus1d(arr, segment_shape):

    indices = np.add.accumulate(segment_shape)[:-1]
    return np.split(arr, indices, axis=0)

class SnippetsError(Exception):
    def __init__(self, result_segment_shapes, message="No valid snippets"):
        self.result_segment_shapes = result_segment_shapes
        self.message = message
        super().__init__(self.message)



def same_segment_with_kaggle(signal,fs,seq_length=50):
    # middle 50s
    if signal.shape[1] / fs < seq_length:
        return None

    elif signal.shape[1] / fs > seq_length:
        seq_samples = seq_length * fs

        start_idx = (signal.shape[1] - seq_samples) // 2
        end_idx = start_idx + seq_samples

        return specify_segment_for_continuous_test(signal,start_idx,end_idx)
    else:
        return signal


def specify_segment_for_continuous_test(signal, start, end):
    return  signal[:, int(start):int(end)]



def recursive_files(root_dir,file_type):
    root_path = Path(root_dir)
    return [file.name for file in root_path.rglob(f'*{file_type}')]

def find_file_path(root_dir, target_filename):
    root_path = Path(root_dir)
    for file in root_path.rglob(target_filename):
        return str(file)  # 返回完整路径
    return None  # 如果未找到文件，返回 None


def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')


