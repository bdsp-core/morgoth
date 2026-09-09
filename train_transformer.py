import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import math
import sys
from typing import Iterable
from einops import rearrange
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F
from timm.models import create_model
import utils
from utils import MorgothNativeScalerWithGradNormCount as NativeScaler
from utils import create_optimizer
import backbone
import tokenizer
from train_tokenizer import augment_eeg

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask.to(torch.bool)


def train_one_epoch(model: torch.nn.Module, vqnsp: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    metric_logger = utils.MorgothMetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.MorgothSmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.MorgothSmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0:
            continue
        input_chans = utils.morgoth_get_input_chans(ch_names)
        for step, (batch) in enumerate(
                metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            it = start_steps + step + step_loader
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples_raw = batch.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples_raw, 'B N (A T) -> B N A T', T=200)

            objective = getattr(args, 'objective', 'mem')
            mlm_acc = mlm_acc_sym = None

            if objective == 'causal':
                my_context = model.no_sync if args.distributed and (
                            step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        input_ids = vqnsp.get_codebook_indices(samples, input_chans)
                with my_context():
                    with torch.cuda.amp.autocast():
                        logits = model(samples, input_chans)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))
                loss_rec = loss

            else:
                bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=args.mask_ratio).to(
                    device, non_blocking=True)

                if objective == 'mae':
                    my_context = model.no_sync if args.distributed and (
                                step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
                    with my_context():
                        with torch.cuda.amp.autocast():
                            pred, target = model(samples, input_chans, bool_masked_pos=bool_masked_pos)
                            loss = F.mse_loss(pred, target.type_as(pred))
                    loss_rec = loss

                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input_ids = vqnsp.get_codebook_indices(samples, input_chans)
                        labels = input_ids[bool_masked_pos]
                        labels_sym = input_ids[~bool_masked_pos]

                    my_context = model.no_sync if args.distributed and (
                                step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
                    with my_context():
                        with torch.cuda.amp.autocast():
                            if objective == 'contrastive':
                                samples_aug = rearrange(augment_eeg(samples_raw), 'B N (A T) -> B N A T', T=200)
                                x_rec, x_rec_sym, cont_loss = model(
                                    samples, input_chans, bool_masked_pos=bool_masked_pos, x_aug=samples_aug,
                                    contrastive_temperature=args.contrastive_temperature)
                            else:
                                x_rec, x_rec_sym = model(samples, input_chans, bool_masked_pos=bool_masked_pos)
                                cont_loss = None

                            loss_rec = loss_fn(x_rec, labels)
                            loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
                            loss = loss_rec + loss_rec_sym
                            if cont_loss is not None:
                                loss = loss + args.contrastive_weight * cont_loss

                    mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
                    mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.morgoth_get_rank()}", force=True)

                sys.exit(1)

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= args.gradient_accumulation_steps
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            if mlm_acc is not None:
                metric_logger.update(mlm_acc=mlm_acc)
                metric_logger.update(mlm_acc_sym=mlm_acc_sym)
            metric_logger.update(loss_rec=loss_rec.item() / (2 if mlm_acc is not None else 1))

            if log_writer is not None:
                if mlm_acc is not None:
                    log_writer.update(mlm_acc=mlm_acc, head="loss")
                    log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
                log_writer.update(loss_rec=loss_rec.item() / (2 if mlm_acc is not None else 1), head="loss")

            metric_logger.update(loss=loss_value)
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
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=args.codebook_size
    )

    return model


def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
        args.tokenizer_model,
        pretrained=True,
        pretrained_weight=args.tokenizer_weight,
        as_tokenzer=True,
        n_code=args.codebook_size,
        code_dim=args.codebook_dim,
    ).eval()
    return model



def get_args():
    parser = argparse.ArgumentParser('EEGfounder pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)

    parser.add_argument("--tokenizer_weight", type=str)
    parser.add_argument("--tokenizer_model", type=str, default="morgoth_tokenizer_base")

    parser.add_argument('--model', default='morgoth_pretrain_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--objective', default='mem', type=str, choices=['mem', 'mae', 'contrastive', 'causal'],
                        help="pretraining objective: 'mem' (default, original BEiT-style masked VQ-token "
                             "classification), 'mae' (direct masked-signal regression), 'contrastive' "
                             "(mem + NT-Xent on two augmented views), 'causal' (GPT-style next-patch prediction)")
    parser.add_argument('--contrastive_weight', default=0.1, type=float,
                        help="NT-Xent loss weight, only used when --objective contrastive")
    parser.add_argument('--contrastive_temperature', default=0.1, type=float,
                        help="NT-Xent temperature, only used when --objective contrastive")
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help="masking ratio for the 'mem'/'mae'/'contrastive' objectives (unused for 'causal')")
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_true', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=1600, type=int,
                        help='EEG input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=64, type=int, help='number of codebook')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)

    return parser.parse_args()



def main(args):
    utils.morgoth_init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.morgoth_get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    datasets_train = [
        ["/data/routine_30_minutes_eeg_edf_hdf5/dataset.hdf5"]
    ]
    time_window = [
        10
    ]
    dataset_train_list, train_ch_names_list = utils.morgoth_build_pretraining_dataset(datasets_train, time_window,
                                                                              stride_size=200*10, start_percentage=0,
                                                                              end_percentage=1)
    vqnsp = get_visual_tokenizer(args).to(device)

    if True:
        num_tasks = utils.morgoth_get_world_size()
        global_rank = utils.morgoth_get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = sum(
            [len(dataset) for dataset in dataset_train_list]) // args.batch_size // num_tasks

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.MorgothTensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_train_list.append(data_loader_train)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    print("Tokenizer = %s" % str(vqnsp))
    total_batch_size = args.batch_size * utils.morgoth_get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.morgoth_cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.morgoth_cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.morgoth_auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, vqnsp, data_loader_train_list,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            ch_names_list=train_ch_names_list,
            args=args,
        )
        if args.output_dir:
            utils.morgoth_save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.morgoth_is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)

