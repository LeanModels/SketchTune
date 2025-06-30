import math
import os
import pickle
from pathlib import Path

import torch
from torch import nn

from kernels import _gather, _scatter_add


class Sub4BitMatMul(torch.autograd.Function):
    @staticmethod
    def forward(quant_grid, codes, X):
        W = _gather(quant_grid, codes)
        return torch.matmul(X, W.t())

    @staticmethod
    def setup_context(ctx, inputs, _):
        quant_grid, codes, X = inputs
        ctx.save_for_backward(quant_grid, codes, X)

    @staticmethod
    def backward(ctx, grad_output):
        quant_grid, codes, X = ctx.saved_tensors
        bits = int(math.log2(quant_grid.shape[-1]))

        W = _gather(quant_grid, codes)

        grad_W = torch.matmul(grad_output.transpose(-2, -1), X)
        while grad_W.dim() > 2:
            grad_W = grad_W.sum(dim=0)
        grad_quant_grid = _scatter_add(grad_W, codes, bits=bits)
        
        grad_X = torch.matmul(grad_output, W)

        return grad_quant_grid, None, grad_X


class Sub4BitLinear(nn.Module):
    def __init__(self, quant_grid, weight_codes, dtype=torch.float16):
        super().__init__()
        if quant_grid.dim() == 2:
            quant_grid = quant_grid.unsqueeze(1)
        if weight_codes.dim() == 2:
            weight_codes = weight_codes.unsqueeze(1)
        self.quant_grid = nn.Parameter(quant_grid.to(dtype))
        self.register_buffer('weight_codes', weight_codes)

    def forward(self, x):
        return Sub4BitMatMul.apply(self.quant_grid, self.weight_codes, x)
    

def dequantize(quant_grid, weight_codes, dtype=torch.bfloat16, device='cuda', target_device='cpu'):
    if quant_grid.dim() == 2:
        quant_grid = quant_grid.unsqueeze(1)
    if weight_codes.dim() == 2:
        weight_codes = weight_codes.unsqueeze(1)

    device = torch.device(device)
    target_device = torch.device(target_device)
    W = _gather(quant_grid.to(dtype).to(device), weight_codes.to(device))
    return W.to(target_device)

def replace_layers(module, quantizers, quant_grids={}, name='', fast=False):
    if isinstance(module, Sub4BitLinear):
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers.keys():
            if not fast:
                delattr(module, attr)
                setattr(module, attr, Sub4BitLinear(
                    quant_grids[name1] if name1 in quant_grids else quantizers[name1][0],
                    quantizers[name1][1],
                    next(tmp.parameters()).dtype
                ))
                del tmp
            else:
                tmp.weight.data = dequantize(quantizers[name1][0], quantizers[name1][1], dtype=next(tmp.parameters()).dtype)

    for name1, child in module.named_children():
        replace_layers(child, quantizers, quant_grids, name + '.' + name1 if name != '' else name1, fast=fast)


def capture_quant_grids(module, quant_grids, name=''):
    if isinstance(module, Sub4BitLinear):
        quant_grids[name] = module.quant_grid.data.cpu()

    for name1, child in module.named_children():
        capture_quant_grids(child, quant_grids, name + '.' + name1 if name != '' else name1)


def save_pretrained(self, output_dir, *args, **kwargs):
    quant_grids = {}
    capture_quant_grids(self, quant_grids)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, 'sketched_params.pkl'), 'wb') as save_file:
        pickle.dump(quant_grids, save_file)