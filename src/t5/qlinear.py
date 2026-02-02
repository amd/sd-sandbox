#
# Copyright (C) 2023-2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.
#

import gc
from collections import defaultdict

import atom
import torch
from torch import Tensor


class AIEGEMM:
    single_aiegemm = None
    gemm_torch = 0

    @classmethod
    def select_op_handle(cls):
        if AIEGEMM.single_aiegemm is None:
            AIEGEMM.gemm_torch = 1
            AIEGEMM.single_aiegemm = atom.atom_npu_gemm("bfloat16", "uint4", "bfloat16")
        else:
            pass  # single object
        print(f"{AIEGEMM.single_aiegemm=}")

    @classmethod
    def delete(cls):
        del AIEGEMM.single_aiegemm
        AIEGEMM.single_aiegemm = None
        print(f"{AIEGEMM.single_aiegemm=}")


class QLinearPerGrp(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    wts_cnt = 0
    single_aiegemm = atom.atom_npu_gemm("bfloat16", "uint4", "bfloat16")

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = False,
        device=None,
        w_bit: int = 4,
        group_size: int = 128,
        profiler: bool = False,
        model_name="",
        pickle: bool = False,
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.weight = None
        self.qweight = None
        self.qzeros = None
        self.scales = None
        self.profiler = profiler
        self.wts_index = QLinearPerGrp.wts_cnt
        self.model_name = model_name
        self.biasexists = None
        self.weights_quantized = False
        self.pickle = pickle
        if pickle:
            self.aiegemm = QLinearPerGrp.single_aiegemm

    def __repr__(self):
        if self.biasexists is None:
            self.biasexists = True if self.bias is not None else False
        return f"ryzenAI.QLinearPerGrp(in_features:{self.in_features}, out_features:{self.out_features}, bias:{self.biasexists}, device:{self.device}, w_bit:{self.w_bit}, model_name:{self.model_name}, group_size:{self.group_size} )"

    @torch.no_grad()
    def pack(self, qw):
        # this supports 4bit packing
        import math

        qcompact = torch.empty(
            qw.shape[0], math.ceil(qw.shape[1] / 2), dtype=torch.uint8
        )
        j = 0
        for i in range(qw.shape[1]):
            if i % 2 == 0:
                qcompact[:, j] = qw[:, i]
                qcompact[:, j] = qcompact[:, j] << 4
            else:
                qcompact[:, j] = torch.bitwise_or(qcompact[:, j], qw[:, i])
                j += 1
        return qcompact

    @torch.no_grad()
    def unpack(self, qcompact, k):
        # this supports 4bit unpacking
        if qcompact.shape[1] == k:
            return qcompact
        else:
            qw = torch.empty((qcompact.shape[0], k), dtype=torch.int8)
            refmsb = torch.tensor(0xF0, dtype=torch.uint8)
            reflsb = torch.tensor(0x0F, dtype=torch.uint8)
            qw[:, 0::2] = (torch.bitwise_and(qcompact[:, :], refmsb) >> 4).to(
                torch.int8
            )
            qw[:, 1::2] = torch.bitwise_and(qcompact[:, :], reflsb).to(torch.int8)
            return qw

    @torch.no_grad()
    def quantize_weights(self):
        self.weights_quantized = True
        if (self.qweight is None) and (self.weight is not None):  # pergrp
            self.w_shape_orig = self.weight.shape
            w = self.weight.reshape(-1, self.group_size)

            # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)

            # Calculate the scale factor and zero point.
            max_int = 2**self.w_bit - 1
            self.scales = ((max_val - min_val).clamp(min=1e-5) / max_int).to(
                torch.bfloat16
            )
            assert self.scales.shape == max_val.shape
            self.qzeros = (
                (-torch.round(min_val / self.scales)).clamp_(0, max_int).to(torch.int8)
            )
            assert self.scales.shape == min_val.shape

            assert torch.isnan(self.scales).sum() == 0
            assert torch.isnan(w).sum() == 0

            # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
            self.qweight = torch.clamp(
                torch.round(w / self.scales) + self.qzeros, 0, max_int
            ).to(torch.int8)

            assert (
                self.qweight.dim() == 2
                and self.qweight.size(0) == self.scales.size(0)
                and self.qweight.size(1) == self.group_size
            )

            self.qweight = self.qweight.reshape(self.w_shape_orig).to(torch.int8)
            self.qzeros = self.qzeros.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            ).to(torch.int8)
            self.scales = self.scales.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            )
            del self.weight, max_val, min_val, w
            self.wshape = self.qweight.shape
            self.qweight = self.pack(self.qweight)
            self.qzeros.requires_grad_(False)
            self.qweight.requires_grad_(False)
            self.scales.requires_grad_(False)
            gc.collect()
        else:
            print("Skipping - weights already quantized for this layer.")

    def initialize_parameters(self):
        if self.bias is not None:
            self.bias.data = self.bias.to(torch.bfloat16).to(torch.float32)
            self.biasexists = "True"
        else:
            self.bias = torch.zeros((self.out_features), dtype=torch.float32)
            self.biasexists = "False"

        if self.weights_quantized is True:
            self.qweight = self.unpack(
                self.qweight, self.qzeros.shape[1] * self.group_size
            )

            if self.device == "aie":
                self.qweight = self.qweight.transpose(0, 1)
                self.qzeros = self.qzeros.transpose(0, 1)
                self.scales = self.scales.to(torch.float).transpose(0, 1)
                self.wts_index = QLinearPerGrp.wts_cnt
                if self.pickle:
                    nodes = self.aiegemm.initialize_params(
                        self.qweight,
                        self.qzeros,
                        self.scales,
                        self.bias,
                        self.group_size,
                        dict(),
                        self.wts_index,
                    )
                else:
                    nodes = QLinearPerGrp.single_aiegemm.initialize_params(
                        self.qweight,
                        self.qzeros,
                        self.scales,
                        self.bias,
                        self.group_size,
                        dict(),
                        self.wts_index,
                    )
                QLinearPerGrp.wts_cnt += nodes

                self.c_token = torch.zeros(1, self.out_features, dtype=torch.bfloat16)

                if self.pickle:
                    self.forward_dict_aie_mladf = defaultdict(
                        self._get_forward_aie_prefill_mladf2,
                        {1: self.forward_aie_token_mladf2},
                    )
                else:
                    self.forward_dict_aie_mladf = defaultdict(
                        lambda: self.forward_aie_prefill_mladf,
                        {1: self.forward_aie_token_mladf},
                    )
                self.forward_dict_aie = self.forward_dict_aie_mladf
                self.forward_func = self.forward_aie
                del self.qweight, self.qzeros, self.scales, self.bias

            else:  # cpu
                self.weight = self.qweight - torch.repeat_interleave(
                    self.qzeros, self.group_size, dim=1
                )
                self.weight = self.weight * torch.repeat_interleave(
                    self.scales, self.group_size, dim=1
                )
                self.weight = self.weight.transpose(0, 1).to(torch.bfloat16)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(torch.bfloat16)
                self.forward_func = self.forward_cpu
                del self.qweight, self.qzeros, self.scales
        else:  # always on CPU
            self.weight.data = self.weight.data.transpose(0, 1).to(torch.bfloat16)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(torch.bfloat16)
            self.forward_func = self.forward_cpu

        gc.collect()

    def forward_cpu(self, x: Tensor) -> Tensor:
        x = torch.matmul(x.to(torch.bfloat16), self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def forward_aie_token_mladf(self, x: Tensor) -> Tensor:
        QLinearPerGrp.single_aiegemm.execute_aie(x, self.c_token, self.wts_index)
        return self.c_token

    def forward_aie_prefill_mladf(self, x: Tensor) -> Tensor:
        c = torch.zeros((x.shape[0], self.out_features), dtype=torch.bfloat16)
        QLinearPerGrp.single_aiegemm.execute_aie(x, c, self.wts_index)
        return c

    def forward_aie_token_mladf2(self, x: Tensor) -> Tensor:
        x = x.to(torch.bfloat16)
        return self.aiegemm.execute_2_aie_bo(x, self.wts_index)

    def forward_aie_prefill_mladf2(self, x: Tensor) -> Tensor:
        x = x.to(torch.bfloat16)
        return self.aiegemm.execute_2_aie_bo(x, self.wts_index)

    def _get_forward_aie_prefill_mladf2(self):
        return self.forward_aie_prefill_mladf2

    def forward_aie(self, x: Tensor) -> Tensor:
        return self.forward_dict_aie[x.shape[0]](x)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            has_batch = True
        else:
            x = x.unsqueeze(0)
            has_batch = False
        y = torch.empty(
            (x.shape[0], x.shape[1], self.out_features), dtype=torch.bfloat16
        )
        for i in range(x.shape[0]):
            y[i] = self.forward_func(x[i])
        if has_batch is False:
            return y.squeeze(0)
        else:
            return y
