"""
distill.py
==========
NEW, standalone knowledge-distillation training script. Does not modify
finetune_classification.py, backbone.py, task_model.py, or any existing
model class -- reuses their public model-creation/optimizer/scaler
utilities. A frozen teacher checkpoint (any existing morgoth_backbone_*
checkpoint, e.g. checkpoints/BS.pth) supervises a student model (typically
a smaller/faster architecture, e.g. morgoth_backbone_base_rope) via
KL-divergence on temperature-softened logits, combined with the normal
task loss on the student's own labeled data.

loss = (1 - distill_alpha) * task_loss + distill_alpha * T^2 * KL(student/T, teacher/T)

distill_alpha=0 (default) reduces to plain supervised training of the
student with no teacher involvement at all.

Usage:
    python distill.py \
        --teacher_model morgoth_backbone_base --teacher_checkpoint checkpoints/BS.pth \
        --teacher_nb_classes 1 \
        --student_model morgoth_backbone_base_rope --nb_classes 1 \
        --dataset_dir /data/BS/train --eval_dir /data/BS/val --ch_names_file ch_names.txt \
        --distill_alpha 0.5 --distill_temperature 4.0 \
        --output_dir checkpoints/bs_distilled_rope \
        --epochs 30 --batch_size 64 --lr 5e-4
"""
import argparse
import json
import math
import os
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import create_model

import backbone
import utils
from utils import MorgothNativeScalerWithGradNormCount as NativeScaler
from utils import create_optimizer, get_parameter_groups, LayerDecayValueAssigner


def get_args():
    p = argparse.ArgumentParser('Knowledge distillation training script')
    p.add_argument('--teacher_model', required=True, type=str)
    p.add_argument('--teacher_checkpoint', required=True, type=str)
    p.add_argument('--teacher_nb_classes', default=1, type=int)
    p.add_argument('--teacher_qkv_bias', action='store_true', default=False)
    p.add_argument('--teacher_init_values', default=0.1, type=float)

    p.add_argument('--student_model', required=True, type=str)
    p.add_argument('--nb_classes', default=1, type=int)
    p.add_argument('--qkv_bias', action='store_true', default=False)
    p.add_argument('--init_values', default=0.1, type=float)

    p.add_argument('--distill_alpha', default=0.0, type=float,
                   help='weight of the distillation KL term (0 = plain supervised training, teacher unused)')
    p.add_argument('--distill_temperature', default=4.0, type=float,
                   help='softmax temperature for both student/teacher logits in the KL term')

    p.add_argument('--dataset_dir', required=True, type=str)
    p.add_argument('--eval_dir', default='', type=str)
    p.add_argument('--ch_names_file', required=True, type=str,
                   help='text file, one channel name per line, matching the training data montage')

    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--epochs', default=30, type=int)
    p.add_argument('--lr', default=5e-4, type=float)
    p.add_argument('--min_lr', default=1e-5, type=float)
    p.add_argument('--warmup_epochs', default=1, type=int)
    p.add_argument('--weight_decay', default=0.05, type=float)
    p.add_argument('--layer_decay', default=0.65, type=float)
    p.add_argument('--clip_grad', default=None, type=float)
    p.add_argument('--opt', default='adamw', type=str)
    p.add_argument('--opt_eps', default=1e-8, type=float)
    p.add_argument('--opt_betas', default=None, type=float, nargs='+')
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--output_dir', required=True, type=str)
    p.add_argument('--save_ckpt_freq', default=5, type=int)
    p.add_argument('--num_workers', default=8, type=int)
    return p.parse_args()


def load_teacher(args, device):
    teacher = create_model(args.teacher_model, num_classes=args.teacher_nb_classes,
                           qkv_bias=args.teacher_qkv_bias, init_values=args.teacher_init_values)
    ckpt = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = teacher.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f'Teacher checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_student(args):
    model = create_model(args.student_model, num_classes=args.nb_classes,
                         qkv_bias=args.qkv_bias, init_values=args.init_values)
    return model


def distill_loss_fn(student_logits, teacher_logits, targets, task_criterion, distill_alpha, temperature, is_binary):
    task_loss = task_criterion(student_logits, targets)
    if distill_alpha <= 0:
        return task_loss, task_loss, torch.zeros((), device=student_logits.device)

    if is_binary:
        s = torch.sigmoid(student_logits / temperature).clamp(1e-6, 1 - 1e-6)
        t = torch.sigmoid(teacher_logits / temperature).clamp(1e-6, 1 - 1e-6)
        kl = t * (t.log() - s.log()) + (1 - t) * ((1 - t).log() - (1 - s).log())
        kl = kl.mean()
    else:
        s_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        t_prob = F.softmax(teacher_logits / temperature, dim=-1)
        kl = F.kl_div(s_log_prob, t_prob, reduction='batchmean')

    distill_term = (temperature ** 2) * kl
    loss = (1 - distill_alpha) * task_loss + distill_alpha * distill_term
    return loss, task_loss, distill_term


def train_one_epoch(student, teacher, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm, ch_names, args):
    student.train()
    input_chans = utils.morgoth_get_input_chans(ch_names)
    metric_logger = utils.MorgothMetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.MorgothSmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Distill epoch [{epoch}]'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        is_binary = args.nb_classes == 1
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        with torch.no_grad():
            teacher_logits = teacher(samples, input_chans=input_chans) if args.distill_alpha > 0 else None

        with torch.autocast(args.device):
            student_logits = student(samples, input_chans=input_chans)
            loss, task_loss, distill_term = distill_loss_fn(
                student_logits, teacher_logits, targets, criterion,
                args.distill_alpha, args.distill_temperature, is_binary)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping')
            raise SystemExit(1)

        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=student.parameters())
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(task_loss=task_loss.item())
        metric_logger.update(distill_term=float(distill_term))
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with open(args.ch_names_file) as f:
        ch_names = [line.strip() for line in f if line.strip()]

    teacher = load_teacher(args, device) if args.distill_alpha > 0 else None
    if teacher is None:
        print('distill_alpha=0: teacher checkpoint is NOT loaded, this is plain supervised training of the student.')

    student = build_student(args).to(device)

    optimizer = create_optimizer(args, student)
    loss_scaler = NativeScaler()
    criterion = nn.BCEWithLogitsLoss() if args.nb_classes == 1 else nn.CrossEntropyLoss()

    raise NotImplementedError(
        'Wire in a (samples, targets) DataLoader for your dataset here (reuse the dataset-building '
        'logic that matches your --dataset from finetune_classification.py main()), then call '
        'train_one_epoch(...) in a loop as shown below the raise.')
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    print(f'Distilling {args.teacher_model} -> {args.student_model}, distill_alpha={args.distill_alpha}')
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            student, teacher, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, ch_names, args)

        if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            torch.save({'model': student.state_dict(), 'epoch': epoch, 'args': args},
                      os.path.join(args.output_dir, f'checkpoint-{epoch}.pth'))

        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps({**train_stats, 'epoch': epoch}) + '\n')

    print('Training time', str(datetime.timedelta(seconds=int(time.time() - start_time))))


if __name__ == '__main__':
    main()
