import torch
from torch import nn
import cupy as cp
import math
from os import path

with open(path.join(path.dirname(__file__), "gather_sub4bit.cu"), "r") as f:
    kernel_code = f.read()
    _gather_sub4bit = cp.RawKernel(
        kernel_code,
        'gather_sub4bit'
    )

with open(path.join(path.dirname(__file__), "scatter_add_sub4bit.cu"), "r") as f:
    kernel_code = f.read()
    _scatter_add_sub4bit = cp.RawKernel(
        kernel_code,
        'scatter_add_sub4bit'
    )

with open(path.join(path.dirname(__file__), "scatter_add_sub4bit_bf16.cu"), "r") as f:
    kernel_code = f.read()
    _scatter_add_sub4bit_bf16 = cp.RawKernel(
        kernel_code,
        'scatter_add_sub4bit_bf16'
    )

def _gather(src: torch.Tensor, codes: torch.Tensor):
    assert src.dtype == torch.float16 or src.dtype == torch.bfloat16
    assert codes.dtype == torch.uint8
    assert src.dim() == codes.dim()

    nrows, ngrids, eb = src.shape
    nrows_, ngrids_, ncols_per_grid = codes.shape
    assert nrows == nrows_
    assert ngrids == ngrids_

    bits = int(math.log2(eb))
    assert bits <= 4
    assert (2 ** bits) == eb
    ncols_per_grid *= 2

    dst = torch.empty(nrows, ncols_per_grid * ngrids, dtype=src.dtype, device=src.device)
    blocks_per_grid = (nrows * ngrids, )

    with cp.cuda.Device(src.device.index):
        threads_per_block = (128, )
        _gather_sub4bit(grid=blocks_per_grid, block=threads_per_block, shared_mem=2 ** bits * 2 + threads_per_block[0] * 16, args=[
            src.data_ptr(), codes.data_ptr(), dst.data_ptr(), bits, ncols_per_grid,
        ])

    return dst

def _scatter_add(src, codes, bits=4):
    assert src.dtype == torch.float16 or src.dtype == torch.bfloat16
    assert codes.dtype == torch.uint8
    assert bits <= 4

    nrows, ncols = src.shape
    nrows_, ngrids, ncols_per_grid = codes.shape
    ncols_per_grid *= 2

    assert nrows == nrows_
    assert ncols == ncols_per_grid * ngrids

    dst = torch.empty(nrows, ngrids, 2 ** bits, dtype=src.dtype, device=src.device)
    blocks_per_grid = (nrows * ngrids, )

    with cp.cuda.Device(src.device.index):
        threads_per_block = (16, )
        if src.dtype == torch.float16:
            _scatter_add_sub4bit(grid=blocks_per_grid, block=threads_per_block, shared_mem=2 ** bits * 4, args=[
                src.data_ptr(), codes.data_ptr(), dst.data_ptr(), bits, ncols_per_grid,
            ])
        else:
            _scatter_add_sub4bit_bf16(grid=blocks_per_grid, block=threads_per_block, shared_mem=2 ** bits * 4 * threads_per_block[0], args=[
                src.data_ptr(), codes.data_ptr(), dst.data_ptr(), bits, ncols_per_grid,
            ])

    return dst
