import math
import sys
from typing import Iterable
import torch
import torch.backends.cudnn as cudnn
import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path
import tokenizer
from timm.models import create_model
import utils
from utils import MorgothNativeScalerWithGradNormCount as NativeScaler
from utils import create_optimizer


def train_one_epoch(model: torch.nn.Module,
                    data_loader_list: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    clip_grad: float = 0,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    ch_names_list=None,
                    args=None,
                    ):
    model.train()
    metric_logger = utils.MorgothMetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.MorgothSmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.MorgothSmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before each epoch")
        except:
            pass
    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        input_chans = utils.morgoth_get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            it = start_steps + step + step_loader
            if lr_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
            EEG = batch.float().to(device, non_blocking=True) / 100

            EEG_aug = augment_eeg(EEG) if args.contrastive_weight > 0 else None

            with torch.cuda.amp.autocast(enabled=True):
                loss, log_loss = model(
                    EEG, input_chans=input_chans, x_aug=EEG_aug,
                    contrastive_weight=args.contrastive_weight,
                    contrastive_temperature=args.contrastive_temperature,
                )

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), force=True)
                utils.save_nan_model(args, model)
                sys.exit(1)

            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            new_log_loss = {k.split('/')[-1]: v for k, v in log_loss.items() if k not in ['total_loss']}
            metric_logger.update(**new_log_loss)

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
                log_writer.update(**new_log_loss, head="train/loss")

                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.update(loss_scale=loss_scale_value, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['Unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return train_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader_list, model, device, log_writer=None, epoch=None, ch_names_list=None, args=None):
    metric_logger = utils.MorgothMetricLogger(delimiter="  ")
    header = 'Validation:'

    model.eval()

    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before testing")
        except:
            pass

    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        input_chans = utils.morgoth_get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            images = batch.float().to(device, non_blocking=True) / 100
            loss, log_loss = model(images, input_chans=input_chans)

            metric_logger.update(loss=loss.item())

            new_log_loss = {k.split('/')[-1]: v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if hasattr(model, 'module') and hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return test_stat

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def calculate_codebook_usage(data_loader, model, device, log_writer=None, epoch=None, args=None):
    metric_logger = utils.MorgothMetricLogger(delimiter="  ")
    header = 'Calculating codebook usage:'

    model.eval()

    codebook_num = args.codebook_n_emd
    codebook_cnt = torch.zeros(codebook_num, dtype=torch.float64).to(device)

    for step, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.float().to(device, non_blocking=True) / 100

        outputs = utils.morgoth_get_model(model).get_tokens(images)['token'].view(-1)

        outputs_gather_list = [torch.zeros_like(outputs) for _ in range(utils.morgoth_get_world_size())]
        torch.distributed.all_gather(outputs_gather_list, outputs)
        all_tokens = torch.cat(outputs_gather_list, dim=0).view(-1)

        codebook_cnt += torch.bincount(all_tokens, minlength=codebook_num)

    zero_cnt = (codebook_cnt == 0).sum()
    print(f"STAT:  {zero_cnt} tokens ({(zero_cnt / codebook_num) * 100}%) never are used in this codebook.")

def get_model(args, **kwargs):
    model = create_model(
        args.model,
        pretrained=False,
        as_tokenzer=False,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        EEG_size=args.input_size,
        decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
        use_contrastive=(args.contrastive_weight > 0),
    )
    return model


def augment_eeg(
    x: torch.Tensor,
    noise_std: float = 0.05,
    shift_max: int = 50,
    scale_lo: float = 0.8,
    scale_hi: float = 1.2,
) -> torch.Tensor:
    """
    Stochastic augmentation used to build the contrastive positive pair.
    x : [B, N_ch, T]  -- same pre-scaled signal passed into the model.
    Only called when --contrastive_weight > 0.

    Three independent perturbations applied in sequence:
      1. Additive Gaussian noise scaled to per-sample std
      2. Uniform amplitude scaling
      3. Circular time-domain shift
    """
    B, N, T = x.shape
    sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x = x + torch.randn_like(x) * (noise_std * sigma)
    gain = x.new_empty(B, 1, 1).uniform_(scale_lo, scale_hi)
    x = x * gain
    delta = int(torch.randint(-shift_max, shift_max + 1, (1,)).item())
    x = torch.roll(x, delta, dims=-1)
    return x


def get_args():
    parser = argparse.ArgumentParser('Tokenizer training script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--model', default='morgoth_tokenizer_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--codebook_n_emd', default=8192, type=int, metavar='MODEL',
                        help='number of codebook')
    parser.add_argument('--codebook_emd_dim', default=64, type=int, metavar='MODEL',
                        help='number of codebook')
    parser.add_argument('--ema_decay', default=0.99, type=float, metavar='MODEL', help='ema decay for quantizer')
    parser.add_argument('--quantize_kmeans_init', action='store_true', help='enable kmeans_init for quantizer')

    parser.add_argument('--input_size', default=1600, type=int, help='EEG input size for backbone')

    parser.add_argument('--contrastive_weight', default=0.0, type=float,
                        help='InfoNCE/NT-Xent loss weight added on top of the reconstruction losses '
                             '(0 = disabled, matches original behavior)')
    parser.add_argument('--contrastive_temperature', default=0.1, type=float,
                        help='temperature for the InfoNCE/NT-Xent loss, only used when --contrastive_weight > 0')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
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

    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False, help="Perform evaluation only")
    parser.add_argument('--calculate_codebook_usage', action='store_true', default=False)

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

    datasets_train = [
        ["/data/routine_30_minutes_eeg_edf_hdf5/dataset.hdf5"]
    ]
    time_window = [
        10
    ]
    dataset_train_list, train_ch_names_list = utils.morgoth_build_pretraining_dataset(datasets_train, time_window,
                                                                              stride_size=200)

    datasets_val = [
        ["/data/routine_30_minutes_eeg_edf_hdf5/dataset.hdf5"]
    ]
    if args.disable_eval:
        dataset_val_list = None
    else:
        dataset_val_list, val_ch_names_list = utils.morgoth_build_pretraining_dataset(datasets_val, [4])

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
        sampler_eval_list = []
        if args.dist_eval:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                sampler_eval_list.append(sampler_val)
        else:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.SequentialSampler(dataset)
                sampler_eval_list.append(sampler_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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

    if dataset_val_list is not None:
        data_loader_val_list = []
        for dataset, sampler in zip(dataset_val_list, sampler_eval_list):
            data_loader_val = torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
            data_loader_val_list.append(data_loader_val)
    else:
        data_loader_val_list = None

    model.to(device)
    model_without_ddp = model
    if not args.eval:
        print("Model = %s" % str(model_without_ddp))
    for part in ['encoder', 'decoder']:
        model_part = eval(f"model.{part}")
        n_learnable_parameters = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        n_fix_parameters = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
        print(f'number of learnable params in model.{part}: {n_learnable_parameters / 1e6} M')
        print(f'number of fixed params in model.{part}: {n_fix_parameters / 1e6} M')

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    print(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    total_batch_size = args.batch_size * utils.morgoth_get_world_size()
    args.lr = total_batch_size / 128 * args.lr
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weigth Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.morgoth_cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.morgoth_auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, log_writer, 0, args=args)
        exit(0)

    if args.calculate_codebook_usage:
        test_stats = calculate_codebook_usage(data_loader_val, model, device, log_writer, 0, args=args)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train_list,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            ch_names_list=train_ch_names_list,
            args=args
        )
        if args.output_dir:
            utils.morgoth_save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val_list is not None:
            test_stats = evaluate(data_loader_val_list, model, device, log_writer, epoch,
                                  ch_names_list=val_ch_names_list, args=args)
            print(
                f"Validation loss of the network on the {sum([len(dataset) for dataset in dataset_val_list])} test EEG: {test_stats['loss']:.4f}")

            if log_writer is not None:
                log_writer.update(**test_stats, head="val/loss")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_learnable_parameters}

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

