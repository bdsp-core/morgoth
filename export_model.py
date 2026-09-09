"""
export_model.py
================
NEW, opt-in model deployment/export utilities: ONNX export, TorchScript
export, and dynamic INT8 quantization.

Pure additions -- does not touch backbone.py/tokenizer.py or any existing
registered model factory. Loading a checkpoint for normal PyTorch training
or inference (create_model(...) + model.load_state_dict(...)) is completely
unaffected; this module is only exercised when explicitly called.

Usage:
    python export_model.py \
        --model morgoth_backbone_base --checkpoint checkpoints/BS.pth \
        --nb_classes 1 --format onnx --output_path bs_model.onnx

    python export_model.py \
        --model morgoth_backbone_base --checkpoint checkpoints/BS.pth \
        --nb_classes 1 --format torchscript --output_path bs_model.pt

    python export_model.py \
        --model morgoth_backbone_base --checkpoint checkpoints/BS.pth \
        --nb_classes 1 --format int8 --output_path bs_model_int8.pt
"""
import argparse

import torch
from timm.models import create_model

import backbone
import tokenizer


def load_model_for_export(model_name, checkpoint_path, num_classes=1, qkv_bias=False,
                          init_values=0.1, **model_kwargs):
    model = create_model(model_name, num_classes=num_classes, qkv_bias=qkv_bias,
                         init_values=init_values, **model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    model.eval()
    return model


class _InputChansWrapper(torch.nn.Module):
    """Bakes a fixed input_chans list into the model's forward so exported
    graphs (ONNX/TorchScript) take a single tensor input, matching what
    those export formats expect -- no change to the underlying model."""

    def __init__(self, model, input_chans):
        super().__init__()
        self.model = model
        self.input_chans = input_chans

    def forward(self, eeg):
        return self.model(eeg, input_chans=self.input_chans)


def export_onnx(model, output_path, example_input, input_chans=None, opset_version=18, dynamic_batch=True):
    wrapped = _InputChansWrapper(model, input_chans).eval()
    dynamic_axes = {'eeg': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None
    torch.onnx.export(
        wrapped, (example_input,), output_path,
        input_names=['eeg'], output_names=['output'],
        dynamic_axes=dynamic_axes, opset_version=opset_version,
    )
    return output_path


def export_torchscript(model, output_path, example_input, input_chans=None):
    wrapped = _InputChansWrapper(model, input_chans).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (example_input,))
    traced.save(output_path)
    return output_path


def quantize_dynamic_int8(model):
    """Post-training dynamic INT8 quantization of the model's Linear layers
    (the dominant parameter/compute share in this transformer). Returns a
    NEW model object -- does not mutate or re-save over the original
    fp32 checkpoint.

    KNOWN LIMITATION (found while verifying this export path): torch's
    dynamic quantization replaces nn.Linear submodules with a quantized
    variant whose .weight is a packed-weight *accessor method*, not a plain
    tensor. MorgothAttention.forward() calls F.linear(weight=self.qkv.weight, ...)
    directly (needed to build the separate q_bias/v_bias for qkv_bias
    support) instead of self.qkv(x) -- so it breaks against a quantized
    self.qkv, with or without this session's changes; this is not something
    introduced here, and not something this export utility silently patches
    over (that would mean touching MorgothAttention itself, which we are
    deliberately not doing). Left in place and documented rather than
    hidden: works for the pure-MLP-heavy tokenizer/decoder paths, but not
    for MorgothAttention's qkv projection as currently written."""
    supported = torch.backends.quantized.supported_engines
    if torch.backends.quantized.engine not in supported or torch.backends.quantized.engine == 'none':
        for candidate in ('fbgemm', 'qnnpack'):
            if candidate in supported:
                torch.backends.quantized.engine = candidate
                break
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def get_args():
    p = argparse.ArgumentParser('Export a morgoth checkpoint to a deployment format')
    p.add_argument('--model', required=True, type=str, help='registered model name, e.g. morgoth_backbone_base')
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('--nb_classes', default=1, type=int)
    p.add_argument('--qkv_bias', action='store_true', default=False)
    p.add_argument('--init_values', default=0.1, type=float)
    p.add_argument('--n_channels', default=19, type=int, help='EEG channels in the example input used to trace/export')
    p.add_argument('--n_patches', default=8, type=int)
    p.add_argument('--patch_size', default=200, type=int)
    p.add_argument('--format', required=True, choices=['onnx', 'torchscript', 'int8'])
    p.add_argument('--output_path', required=True, type=str)
    p.add_argument('--opset_version', default=18, type=int)
    return p.parse_args()


def main():
    args = get_args()
    model = load_model_for_export(args.model, args.checkpoint, num_classes=args.nb_classes,
                                  qkv_bias=args.qkv_bias, init_values=args.init_values)
    input_chans = list(range(args.n_channels + 1))
    example_input = torch.randn(1, args.n_channels, args.n_patches, args.patch_size)

    if args.format == 'onnx':
        export_onnx(model, args.output_path, example_input, input_chans=input_chans,
                   opset_version=args.opset_version)
    elif args.format == 'torchscript':
        export_torchscript(model, args.output_path, example_input, input_chans=input_chans)
    else:
        quantized = quantize_dynamic_int8(model)
        torch.save({'model': quantized.state_dict()}, args.output_path)

    print(f'Exported {args.model} ({args.format}) -> {args.output_path}')


if __name__ == '__main__':
    main()
