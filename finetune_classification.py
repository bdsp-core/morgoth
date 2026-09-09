import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch.backends.cudnn as cudnn
import json
import os
import pandas as pd
from pathlib import Path
import torch.distributed as dist
from tqdm import tqdm
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils import FocalLoss, DynamicFocalLoss,GHMC,MarginLoss, BinaryFocalLoss
from timm.utils import ModelEma
from utils import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from utils import MorgothNativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
import task_model
import backbone
import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
from timm.utils import ModelEma
import utils
from einops import rearrange
from eeg_augmentations import spec_augment, channel_dropout, time_warp

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('Fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--distributed', default=True, type= bool, help='Enabling distributed')

    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')

    parser.add_argument('--model', default='morgoth_backbone_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--sam_rho', type=float, default=0.0,
                        help='Sharpness-Aware Minimization neighborhood size rho, enabled if > 0 '
                             '(default 0 = disabled, matches original single-step optimizer behavior). '
                             'Only supported in the non-deepspeed (--enable_deepspeed unset) path.')
    parser.add_argument('--sam_adaptive', action='store_true', default=False,
                        help='use Adaptive SAM (ASAM) scaling instead of plain SAM; only used when --sam_rho > 0')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--focalloss', action='store_true', default=False)
    parser.add_argument('--focal_alpha', type=str, default='1.0 1.0 1.0', help='Focal Loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2, help='Focal Loss gamma')

    parser.add_argument('--marginloss', action='store_true', default=False)
    parser.add_argument('--margin', type=float, default=1, help='margin in margin loss')

    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, enabled if > 0 (default 0 = disabled)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, enabled if > 0 (default 0 = disabled)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='probability of applying mixup/cutmix when enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='probability of switching to cutmix when both mixup and cutmix are enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='mixup/cutmix mode: batch, pair, or elem')

    parser.add_argument('--specaugment', action='store_true', default=False,
                        help='enable SpecAugment-style time/sample masking (default: off)')
    parser.add_argument('--specaugment_time_mask', type=int, default=2)
    parser.add_argument('--specaugment_freq_mask', type=int, default=20)
    parser.add_argument('--channel_dropout', type=float, default=0.0,
                        help='per-sample probability of zeroing an entire channel (default 0 = disabled)')
    parser.add_argument('--time_warp', type=float, default=0.0,
                        help='max fractional time-axis stretch/compress (default 0 = disabled)')

    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--task_model', default='',
                        help='resume the fine-tuned task model from checkpoint')

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--train_eeg_montage', default='average',
                        help='average (19) | bipolar (18) | combine (19 average + 18 bipolar)')
    parser.add_argument('--train_eeg_length', default=False, type=int,
                        help='15 | 10 | 5 Training samples length, use it for training IIIC (15s, 10s, 5s), but not for wave like spike. Training sample for IIIC is prepared with 15 seconds, and can be cut for assigned train_eeg_length; Spike samples should be prepared with fixed length like 1 seconds')
    parser.add_argument('--training_data_dir', default='',help='training data dir')

    parser.add_argument('--predict', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--eval_list_file', default='', type=str,
                        help='a file to list test or evaluation dataset')
    parser.add_argument('--eval_list_column', default='',
                        help='file name column in test or evaluation dataset')
    parser.add_argument('--eval_results_dir', default='',
                        help='path where to save prediction results')
    parser.add_argument('--eeg_montage', default='average',
                        help='average (19) | bipolar (18) | combine (19 average + 18 bipolar) | single')

    parser.add_argument('--prediction_slipping_step', default=0, type=int,
                        help='slipping step in continuous prediction, if original hz>200 prediction_slipping_step better to be x/100 or xx/128')

    parser.add_argument('--prediction_slipping_step_second', default=0, type=int,
                        help='slipping step in second in continuous prediction')

    parser.add_argument('--data_format',  default='mat', type=str,
                        help='in continuous test, original data format mat | edf | pkl')

    parser.add_argument('--sampling_rate', default=0, type=int,
                        help='give sampling rate if mat/edf data files do not have this information')

    parser.add_argument('--already_format_channel_order', default='no',
                        choices=['yes','y', 'no','n'],type=str,
                        help='do not have channel info in data file, but the data if ordered by FP1, F3, C3, P3, F7, T3, T5, O1, FZ, CZ, PZ, FP2, F4, C4, P4, F8, T4, T6, O2')

    parser.add_argument('--already_average_montage', default='no',
                        choices=['yes','y', 'no','n'],type=str,
                        help='the original data already have average montage')

    parser.add_argument( '--allow_missing_channels', type=str,
                         choices=['yes','y', 'no','n'], default='no',
                         help='Allow missing channels (yes/no)')
    parser.add_argument('--polarity', default=1, type=int,
                        help='Set 1 ｜ -1, default 1. If set -1, the signal is inverted')

    parser.add_argument('--smooth_result',  default='', type=str,
                        help='smooth the continuous results by EMA')

    parser.add_argument('--need_spikes_10s_result', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='not only output 1-second prediction, but also output 10-second')

    parser.add_argument('--spikes_10s_result_slipping_step_second', default=0, type=int, help='slipping step (in second) in continuous results based on 10s spikes segments. Could set it as 10')

    parser.add_argument('--test_data_format',  default='pkl', type=str,
                        help='in test, original data format mat | pkl')

    parser.add_argument('--max_length_hour', default='no', type=str,
                        help='Only analyze the first n hours of data.')

    parser.add_argument('--rewrite_results', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='rewrite reults')

    parser.add_argument('--leave_one_hemisphere_out', default='no', type=str,
                        choices=['no', 'right', 'left', 'middle'],
                        help='set right or left or middle hemisphere 0')

    parser.add_argument('--channel_symmetric_flip', default='no', type=str,
                        choices=['no', 'right', 'left'],
                        help='symmetrically map the right(left) hemisphere data to the left(right) hemisphere.')


    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: IIIC | SPIKES | NORMAL | SLOWING | BS  | SLEEP')
    parser.add_argument('--eval_sub_dir', type=str,
                        help='dataset: subject directory')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            print("Use deepspeed==0.4.0'")
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(args,model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True,
                    mixup_fn: Optional[Mixup] = None):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.morgoth_get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MorgothMetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.MorgothSmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.MorgothSmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)

        if getattr(args, 'specaugment', False):
            samples = spec_augment(samples, time_mask_param=args.specaugment_time_mask,
                                   freq_mask_param=args.specaugment_freq_mask)
        if getattr(args, 'channel_dropout', 0.0) > 0:
            samples = channel_dropout(samples, drop_prob=args.channel_dropout)
        if getattr(args, 'time_warp', 0.0) > 0:
            samples = time_warp(samples, max_warp=args.time_warp)

        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)
        elif mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        is_sam = isinstance(optimizer, utils.SAM)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        elif is_sam:
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.autocast(args.device):
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        elif is_sam:
            loss.backward()
            optimizer.first_step(zero_grad=True)
            loss2, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            if model_ema is not None:
                model_ema.update(model)
            grad_norm = None
            loss_scale_value = 0.0
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if is_binary:

            class_acc = utils.morgoth_get_metrics(torch.sigmoid(output).detach().cpu().numpy(), (targets >= 0.5).int().detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]

        else:
            if mixup_fn is not None:
                acc_targets = targets.max(-1)[-1]
            else:
                acc_targets = targets.squeeze()
            class_acc = (output.max(-1)[-1] == acc_targets).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def predict(args, data_loader, model, device, header='Prediction:', ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.morgoth_get_input_chans(ch_names)


    model.eval()
    pred = []
    for step, batch in enumerate(data_loader):
        EEG = batch[0]
        EEG = EEG.float().to(device, non_blocking=True)/ 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)

        with torch.autocast(args.device):
            output = model(EEG, input_chans=input_chans)

        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()

        if args.device == 'cpu':
            output = output.to(torch.float32)

        pred.append(output)

    pred = torch.cat(pred, dim=0).numpy()
    return pred

@torch.no_grad()
def evaluate(args, data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=False):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.morgoth_get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MorgothMetricLogger(delimiter="  ")

    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True)
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        with torch.autocast(args.device):
            output = model(EEG, input_chans=input_chans)
            loss = criterion(output, target)
        if is_binary:
            output = torch.sigmoid(output).cpu()
            target=(target >= 0.5).int().detach().cpu()
        else:
            output = output.cpu()
            target = target.cpu()

        results = utils.morgoth_get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.morgoth_get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg

    return ret,pred,true



def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )

    return model


def get_dataset(args, evaluation=False, sub_dir=False):
    Bipolar = False
    addBipolar = False
    sample_length=False
    if args.dataset == 'TUAB' or args.dataset == 'TUEP':
        root=args.training_data_dir
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset == 'TUEV':
        root=args.training_data_dir
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset=='IIIC':
        if args.train_eeg_montage=="combine":
            ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
            addBipolar = True
        elif args.train_eeg_montage == "bipolar":
            ch_names = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
            Bipolar = True

        else:
            ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        sample_length = args.train_eeg_length

    elif args.dataset == "SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root=args.training_data_dir

    elif args.dataset == "FOC_GEN_SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 3
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "FOC_SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root=args.training_data_dir


    elif args.dataset == "GEN_SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir

    elif args.dataset== "SLOWING":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 3
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "NORMAL":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset=='BS':
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir

    elif args.dataset == "MGBSLEEP3stages":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 3
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "SLEEPPSG":
        ch_names = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        root = args.training_data_dir
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "SLEEPPENN":
        ch_names = ['C3','C4']
        root = args.training_data_dir
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "SLEEPMASS":
        ch_names = ['F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    else:
        print ('The dataset in args format is wrong')
        return

    print(f"Data from {root}")

    if evaluation:
        result_file, evaluation_dataset = utils.prepare_classification_dataset(root, only_evaluate=True, original_format=args.test_data_format, sub_dir=sub_dir,Bipolar=Bipolar,addBipolar=addBipolar,sample_length=sample_length)
        if result_file==False:
            return False, False, False,False
        return result_file, evaluation_dataset, ch_names, metrics

    else:
        train_dataset, test_dataset, val_dataset = utils.prepare_classification_dataset(root,Bipolar=Bipolar,addBipolar=addBipolar,sample_length=sample_length)
        return train_dataset, test_dataset, val_dataset, ch_names, metrics


def main(args, ds_init):
    utils.morgoth_init_distributed_mode(args)

    if ds_init is not None:
        utils.morgoth_create_ds_config(args)


    device = torch.device(args.device)

    if args.predict:
        if args.prediction_slipping_step== 0 and args.prediction_slipping_step_second==0:
            print('--prediction_slipping_step or --prediction_slipping_step_second is required.')
            exit(0)

        elif args.prediction_slipping_step != 0:
            used_slipping_step = args.prediction_slipping_step
            step_in_point = True
            if args.prediction_slipping_step_second !=0:
                print('Get both --prediction_slipping_step and --prediction_slipping_step_second, use the first parameter.')

        else:
           used_slipping_step = args.prediction_slipping_step_second
           step_in_point = False


        if str(args.need_spikes_10s_result).lower()=='y' or str(args.need_spikes_10s_result).lower()=='yes':
            args.need_spikes_10s_result=True
            if args.spikes_10s_result_slipping_step_second==0:
                print('Set --need_spikes_10s_result yes, but not set --spikes_10s_result_slipping_step_second or default 10')
                args.spikes_10s_result_slipping_step_second=10
        else:
            args.need_spikes_10s_result=False

        if str(args.already_format_channel_order).lower() == 'yes' or str(args.already_format_channel_order).lower() == 'y':
            args.already_format_channel_order = True
        else:
            args.already_format_channel_order = False

        if str(args.already_average_montage).lower()  == 'yes' or str(args.already_average_montage).lower() == 'y':
            args.already_average_montage = True
        else:
            args.already_average_montage = False

        if str(args.allow_missing_channels).lower()  == 'yes' or str(args.allow_missing_channels).lower() == 'y':
            args.allow_missing_channels = True
        else:
            args.allow_missing_channels = False

        if str(args.leave_one_hemisphere_out).lower()  == 'n' or str(args.leave_one_hemisphere_out).lower() == 'no':
            args.leave_one_hemisphere_out=False
        else:
            args.leave_one_hemisphere_out = str(args.leave_one_hemisphere_out).lower()

        if str(args.channel_symmetric_flip).lower()  == 'n' or str(args.channel_symmetric_flip).lower() == 'no':
            args.channel_symmetric_flip=False
        else:
            args.channel_symmetric_flip = str(args.channel_symmetric_flip).lower()

        if str(args.max_length_hour).lower() == 'n' or str(args.max_length_hour).lower() == 'no':
            args.max_length_hour = None
        else:
            args.max_length_hour = int(args.max_length_hour)


        args.nb_classes = utils.get_n_classes(args.dataset)

        model = get_models(args)

        patch_size = model.patch_size
        args.window_size = (1, args.input_size // patch_size)
        args.patch_size = patch_size

        if args.device == 'cpu':
            model = model.to(torch.float32)

        else:
            model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)


        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            model_without_ddp = model.module

        utils.load_from_task_model(
            args=args, model_without_ddp=model_without_ddp)

        os.makedirs(args.eval_results_dir, exist_ok=True)
        os.chmod(args.eval_results_dir, 0o777)

        if args.eval_list_file:
            if args.eval_list_file.endswith("xlsx"):
                eval_list = pd.read_excel(args.eval_list_file).astype(str)
            elif args.eval_list_file.endswith("csv"):
                eval_list = pd.read_csv(args.eval_list_file).astype(str)
            else:
                print("eval_list_file should be in xlsx or cvs format")
                exit(0)
            file_list=eval_list[args.eval_list_column].to_list()
            file_list = [f"{file.split('.')[0]}.{args.data_format}" for file in file_list]

            current_files = set(os.listdir(args.eval_sub_dir))
            file_list = [file.split('.')[0] for file in file_list if file in current_files]

        else:

            file_list = utils.recursive_files(root_dir=args.eval_sub_dir, file_type=args.data_format)
            file_list = [file.split('.')[0] for file in file_list]
            file_list = sorted(file_list, key=utils.extract_number)

        print(f'\n ')
        print(f'[*] For {args.dataset} task')
        print(f'Input {len(file_list)} EEG files...')

        for eval_file in tqdm(file_list,desc=f"{args.dataset} event level results"):

            if args.rewrite_results in ['no' , 'n' , 'No' , 'N' , 'NO']:
                result_file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                if os.path.exists(result_file_path):
                    print(f'{result_file_path} already exists, skip')
                    continue



            data_path=utils.find_file_path(root_dir=args.eval_sub_dir, target_filename=f'{eval_file}.{args.data_format}')

            if data_path==None:
                print(f"{data_path} data not found")
                continue

            data_transform=utils.Compose([utils.Clipping(),utils.Scaling()])

            if args.eeg_montage == "single":
                df = pd.DataFrame()
                for i in range(19):
                    montage = utils.single_channel_average_montage(i)
                    ch_names=montage.get_channel_names()

                    try:
                        EEG_data = utils.ContinuousToSnippetDataset(
                                            type=args.dataset,
                                            path_signal=data_path,
                                            given_fs=args.sampling_rate,
                                            has_formatted_channel=args.already_format_channel_order,
                                            has_avg=args.already_average_montage,
                                            allow_missing_channels=args.allow_missing_channels,
                                            montage=montage,
                                            transform=data_transform,
                                            step=used_slipping_step,
                                            step_in_point=step_in_point,
                                            polarity=int(args.polarity),
                                            max_length_hour=args.max_length_hour)

                    except ValueError as e:
                        print(f"DataError: {e}")
                        continue

                    except Exception as e:
                        print(f"Unknown error: {e}")
                        continue

                    sampler_evaluate = torch.utils.data.SequentialSampler(EEG_data)
                    data_loader_evaluate = torch.utils.data.DataLoader(
                        EEG_data, sampler=sampler_evaluate,
                        batch_size=int(1.5 * args.batch_size),
                        num_workers=args.num_workers,
                        pin_memory=args.pin_mem,
                        drop_last=False)
                    pred = predict(args, data_loader_evaluate, model, device, header='Prediction:',
                                   ch_names=ch_names,
                                   is_binary=(args.nb_classes == 1))
                    df[f'pred_{ch_names[0]}']= pred.flatten()

                result_file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                df.to_csv(result_file_path, index=False)
                os.chmod(result_file_path, 0o777)

                continue


            elif args.eeg_montage=="combine":
                montage=utils.combine_montage()
                ch_names = montage.get_channel_names()

            elif args.eeg_montage=="bipolar":
                montage = utils.bipolar_montage()
                ch_names = montage.get_channel_names()

            elif args.eeg_montage=="average":
                if args.dataset=='SLEEPPSG':
                    montage = utils.sleep_common_average_montage()
                    ch_names = montage.get_channel_names()
                else:
                    montage = utils.common_average_montage()
                    ch_names=montage.get_channel_names()

            else:
                print(f"The input parameter eeg_montage is set incorrectly")
                continue

            try:
                EEG_data = utils.ContinuousToSnippetDataset(
                                    type=args.dataset,
                                    path_signal=data_path,
                                    given_fs=args.sampling_rate,
                                    has_formatted_channel=args.already_format_channel_order,
                                    has_avg=args.already_average_montage,
                                    allow_missing_channels=args.allow_missing_channels,
                                    montage=montage,
                                    transform=data_transform,
                                    step=used_slipping_step,
                                    step_in_point=step_in_point,
                                    polarity=int(args.polarity),
                                    leave_one_hemisphere_out=args.leave_one_hemisphere_out,
                                    channel_symmetric_flip=args.channel_symmetric_flip,
                                    max_length_hour=args.max_length_hour
                                                              )
            except ValueError as e:
                print(f"DataError: {e}")
                continue

            except utils.SnippetsError as e:
                result_segment_shapes=e.result_segment_shapes
                if args.nb_classes==1:
                    new_result_vector = np.zeros(result_segment_shapes)
                    df = pd.DataFrame({'pred': new_result_vector})
                else:
                    new_shape = (result_segment_shapes, args.nb_classes)
                    new_result_matrix = np.zeros(new_shape)
                    new_result_matrix[..., 0] = 1
                    df = pd.DataFrame(new_result_matrix, columns=[f'class_{i}_prob' for i in range(args.nb_classes)])
                    df['pred_class'] = df.idxmax(axis=1)
                    df['pred_class'] = df['pred_class'].map(lambda x: int(x.split('_')[1]))

                file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                df.to_csv(file_path, index=False)
                os.chmod(file_path, 0o777)
                continue

            except Exception as e:
                print(f"Unknown error: {e}")
                continue

            sampler_evaluate = torch.utils.data.SequentialSampler(EEG_data)

            data_loader_evaluate = torch.utils.data.DataLoader(
                EEG_data, sampler=sampler_evaluate,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False)

            pred = predict(args, data_loader_evaluate, model, device, header='Prediction:',
                                  ch_names=ch_names,
                                  is_binary=(args.nb_classes == 1))

            valid_index, valid_start_end_indices, result_segment_shapes = EEG_data.get_valid_indices()

            if args.nb_classes==1:
                prob=pred.flatten()

                new_result_vector = np.zeros(len(valid_index), dtype=prob.dtype)
                prob_segments = utils.split_nd_to_plus1d(arr=prob, segment_shape=result_segment_shapes)

                for valid_start_end_index, prob_segment in zip(valid_start_end_indices, prob_segments):
                    valid_start_index,valid_end_index=valid_start_end_index
                    valid_result_segment_shape=valid_end_index-valid_start_index+1

                    new_prob_segment=utils.resize_array_along_axis0(arr=prob_segment, d=1,target_length=valid_result_segment_shape)

                    new_result_vector[valid_start_index:valid_end_index+1]=new_prob_segment

                if args.dataset == 'SPIKES':
                    original_fs = EEG_data.get_original_fs()

                    if args.need_spikes_10s_result:
                        if (step_in_point==True and used_slipping_step/original_fs>1 or (step_in_point==False and used_slipping_step!=1)):
                            print('could not output 10s continuous results for spikes, the step in 1s results should <= 1s')
                        else:
                            results_10s=utils.spikes_results_from_1s_to_10s(
                                predictions=new_result_vector,
                                data_fs=original_fs,
                                original_result_step= used_slipping_step, original_step_in_point=step_in_point,new_result_step_second=args.spikes_10s_result_slipping_step_second)

                            df_10s = pd.DataFrame({'pred': results_10s})
                            file_path_10s = os.path.join(args.eval_results_dir, f'{eval_file}_10s.csv')
                            df_10s.to_csv(file_path_10s, index=False)
                            os.chmod(file_path_10s, 0o777)

                    if step_in_point:
                        smooth_method = args.smooth_result
                        new_result_vector = utils.continuous_binary_probabilities(predictions=new_result_vector, data_fs=original_fs,result_step=args.prediction_slipping_step, smooth_method=smooth_method)
                        if args.allow_missing_channels == True:
                            new_result_vector = utils.continuous_binary_probabilities2(
                            predictions=new_result_vector,
                            data_fs=original_fs,
                            result_step=args.prediction_slipping_step)


                df = pd.DataFrame({'pred': new_result_vector})

            else:
                prob = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
                prob = np.array(prob)

                new_shape = (len(valid_index), prob.shape[1])
                new_result_matrix = np.zeros(new_shape, dtype=prob.dtype)
                new_result_matrix[..., 0] = 1

                prob_segments=utils.split_nd_to_plus1d(arr=prob, segment_shape=result_segment_shapes)

                for valid_start_end_index, prob_segment in zip(valid_start_end_indices, prob_segments):
                    valid_start_index,valid_end_index=valid_start_end_index
                    valid_result_segment_shape=valid_end_index-valid_start_index+1

                    new_prob_segment=utils.resize_array_along_axis0(arr=prob_segment, d=2, target_length=valid_result_segment_shape)

                    new_result_matrix[valid_start_index:valid_end_index+1,:]=new_prob_segment

                if args.dataset == 'IIIC':
                    original_fs = EEG_data.get_original_fs()
                    if step_in_point:
                        new_result_matrix=utils.continuous_multiclass_probabilities(prediction_matrix=new_result_matrix, data_fs=original_fs,result_step=args.prediction_slipping_step, use_smooth=True)

                df = pd.DataFrame(new_result_matrix, columns=[f'class_{i}_prob' for i in range(pred.shape[1])])
                df['pred_class'] = df.idxmax(axis=1)
                df['pred_class'] = df['pred_class'].map(lambda x: int(x.split('_')[1]))

            df = df.fillna(1)
            file_path=os.path.join(args.eval_results_dir, f'{eval_file}.csv')
            df.to_csv(file_path, index=False)
            os.chmod(file_path, 0o777)

        print(f'Event level results are saved to {args.eval_results_dir}')
        if dist.is_initialized():
            dist.destroy_process_group()
        exit(0)

    if args.eval:
        print(args)
        all_items = os.listdir(args.eval_sub_dir)

        subdirectories = [os.path.join(args.eval_sub_dir, item) for item in all_items if
                          os.path.isdir(os.path.join(args.eval_sub_dir, item))]

        if len(subdirectories) == 0:
            subdirectories = [args.eval_sub_dir]

        model = get_models(args)

        patch_size = model.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (1, args.input_size // patch_size)
        args.patch_size = patch_size

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

        utils.load_from_task_model(
            args=args, model_without_ddp=model_without_ddp)

        for eval_sub in tqdm(subdirectories):
            result_file, dataset_evaluate, ch_names, metrics = get_dataset(args, evaluation=True, sub_dir=eval_sub)
            if result_file==False:
                continue

            sampler_evaluate = torch.utils.data.SequentialSampler(dataset_evaluate)

            data_loader_evaluate = torch.utils.data.DataLoader(
                dataset_evaluate, sampler=sampler_evaluate,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False)
            balanced_accuracy = []
            accuracy = []
            test_stats, pred, true = evaluate(args, data_loader_evaluate, model, device, header='Evaluate:',
                                              ch_names=ch_names, metrics=metrics,
                                              is_binary=(args.nb_classes == 1))
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
            print(
                f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
            try:
                df = pd.read_csv(result_file)

                if args.nb_classes == 1:
                    df['true'] = true.flatten()
                    df['pred'] = pred.flatten()

                else:
                    df['true'] = true.flatten()
                    probs= np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
                    for i in range(pred.shape[1]):
                        df[f'class_{i}_prob'] = probs[:, i]

                    df['pred_class'] = df[[f'class_{i}_prob' for i in range(pred.shape[1])]].idxmax(axis=1)
                    df['pred_class'] = df['pred_class'].map(lambda x: int(x.split('_')[1]))


                df = df.fillna(1)
                df.to_csv(result_file, index=False)
                os.chmod(result_file, 0o777)
                print(f"Test results saved to {result_file}")

            except pd.errors.EmptyDataError:
                print("The file is empty or has no columns to parse.")
            except FileNotFoundError:
                print(f"The file {result_file} does not exist.")
            except Exception as e:
                print(f"An error occurred: {e}")

        if dist.is_initialized():
            dist.destroy_process_group()
        exit(0)


    print(args)
    seed = args.seed + utils.morgoth_get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if args.distributed:
        num_tasks = utils.morgoth_get_world_size()
        global_rank = utils.morgoth_get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.MorgothTensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            print("Load pretrained model main part (Student), omit lm_head and projection_head")
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[len('student.'):]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.morgoth_load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)


    total_batch_size = args.batch_size * args.update_freq * utils.morgoth_get_world_size()

    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        if args.sam_rho > 0.:
            base_optimizer_cls = type(optimizer)
            sam_kwargs = {}
            if getattr(args, 'opt_eps', None) is not None:
                sam_kwargs['eps'] = args.opt_eps
            if getattr(args, 'opt_betas', None) is not None:
                sam_kwargs['betas'] = tuple(args.opt_betas)
            optimizer = utils.SAM(
                optimizer.param_groups, base_optimizer_cls, rho=args.sam_rho,
                adaptive=args.sam_adaptive, **sam_kwargs)
            print(f"SAM enabled (rho={args.sam_rho}, adaptive={args.sam_adaptive}), "
                  f"base optimizer={base_optimizer_cls.__name__}")
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.morgoth_cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    wd_schedule_values = utils.morgoth_cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        if args.focalloss:
            alpha = list(map(float, args.focal_alpha.split()))[0]
            gamma = args.focal_gamma
            criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)

        else:
            criterion = torch.nn.BCEWithLogitsLoss()

    elif args.focalloss:
        alpha = list(map(float, args.focal_alpha.split()))
        alpha = torch.tensor(alpha).to(device, non_blocking=True)
        gamma=args.focal_gamma
        if gamma==0:
            criterion=DynamicFocalLoss(alpha=alpha,)
        else:
            criterion=FocalLoss(alpha=alpha,gamma=gamma)

    elif args.marginloss:
        margin=args.margin
        criterion = MarginLoss(margin=margin)

    elif args.marginloss:
        criterion=GHMC()

    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    mixup_fn = None
    if (args.mixup > 0. or args.cutmix > 0.):
        if args.nb_classes > 1:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode, label_smoothing=args.smoothing,
                num_classes=args.nb_classes)
            criterion = SoftTargetCrossEntropy()
            print(f"Mixup/CutMix enabled (mixup={args.mixup}, cutmix={args.cutmix}) "
                  f"-- criterion switched to SoftTargetCrossEntropy")
        else:
            print("--mixup/--cutmix requested but nb_classes<=1 (binary task): "
                  "Mixup needs multi-class one-hot targets, skipping -- binary criterion unchanged")

    utils.morgoth_auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(args,
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_classes == 1, mixup_fn=mixup_fn
        )

        if args.output_dir and args.save_ckpt:
            utils.morgoth_save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            val_stats,_,_ = evaluate(args, data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                                 is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats,_,_ = evaluate(args, data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                  is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.morgoth_save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_accuracy_test = test_stats["accuracy"]

            print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.morgoth_is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
