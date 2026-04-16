#

# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#

import base64
import gc
import json
import os
import pickle
from collections import defaultdict

import torch
from torch import Tensor
import atom


class AIEGEMM:
    single_aiegemm = None
    gemm_torch = 0
    op_version = "v1"
    preemption = False
    pickle = False

    @classmethod
    def select_op_handle(cls):
        if AIEGEMM.single_aiegemm is None:
            AIEGEMM.gemm_torch = 1
            AIEGEMM.single_aiegemm = atom.atom_npu_gemm(
                "bfloat16",
                "uint4",
                "bfloat16",
                cls.op_version,
                cls.preemption,
                cls.pickle,
            )
        else:
            pass  # single object

    @classmethod
    def delete(cls):
        del AIEGEMM.single_aiegemm
        AIEGEMM.single_aiegemm = None

    @classmethod
    def load_npu(cls, model, metadata):
        """Restore NPU state from safetensors metadata and wire all QLinearPerGrp layers."""
        if "__npu_state__" not in metadata:
            raise RuntimeError(
                "Checkpoint missing NPU state — re-run transform_gptq.py with --pickle"
            )
        aiegemm = pickle.loads(base64.b64decode(metadata["__npu_state__"]))
        wts_map = json.loads(metadata["__wts_map__"])
        cls.single_aiegemm = aiegemm
        for name, mod in model.named_modules():
            if isinstance(mod, QLinearPerGrp):
                mod.attach_npu(aiegemm, wts_map[name])


class QLinearPerGrp(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    wts_cnt = 0

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
    ) -> None:
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.register_buffer("weight", None)
        self.register_buffer("qweight", None)
        self.register_buffer("qzeros", None)
        self.register_buffer("scales", None)
        self.register_buffer("bias", None)
        self.profiler = profiler
        self.wts_index = QLinearPerGrp.wts_cnt
        self.model_name = model_name
        self.biasexists = None
        self.weights_quantized = False

    @staticmethod
    def prepare_model(model, metadata):
        """Replace Linear layers with QLinearPerGrp based on safetensors metadata."""
        for mod_path, spec in json.loads(metadata["__qlinear__"]).items():
            parent_path, attr = mod_path.rsplit(".", 1)
            node = QLinearPerGrp(
                in_features=spec["in"],
                out_features=spec["out"],
                w_bit=spec["w_bit"],
                group_size=spec["gs"],
            )
            node.weights_quantized = True
            setattr(model.get_submodule(parent_path), attr, node)

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

            self.qweight = self.qweight.reshape(self.w_shape_orig)
            self.qzeros = self.qzeros.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            ).to(torch.int8)
            self.scales = self.scales.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            )
            self.weight = None
            del max_val, min_val, w
            self.wshape = self.qweight.shape
            self.qweight = self.pack(self.qweight)
            self.qzeros.requires_grad_(False)
            self.qweight.requires_grad_(False)
            self.scales.requires_grad_(False)
            gc.collect()
        else:
            print(f"Skipping - weights already quantized for this layer.")

    def _wire_npu(self):
        self.c_token = torch.zeros(1, self.out_features, dtype=torch.bfloat16)
        self.forward_dict_aie_mladf = defaultdict(
            self._default_forward_aie,
            {1: self.forward_aie_token_mladf},
        )
        self.forward_dict_aie = self.forward_dict_aie_mladf
        self.forward_func = self.forward_aie
        self.qweight = None
        self.qzeros = None
        self.scales = None
        self.bias = None

    def attach_npu(self, aiegemm, wts_index):
        self.device = "aie"
        self.aiegemm = aiegemm
        self.wts_index = wts_index
        self.biasexists = "True" if getattr(self, "bias", None) is not None else "False"
        if self.biasexists == "False":
            self.bias = torch.zeros(self.out_features, dtype=torch.float32)
        self._wire_npu()

    def _default_forward_aie(self):
        return (
            self.forward_aie_prefill_mladf_bo_manager
            if AIEGEMM.pickle
            else self.forward_aie_prefill_mladf
        )

    def initialize_parameters(self):
        if self.bias is not None:
            self.bias.data = self.bias.to(torch.bfloat16).to(torch.float32)
            self.biasexists = "True"
        else:
            self.bias = torch.zeros((self.out_features), dtype=torch.float32)
            self.biasexists = "False"

        if self.weights_quantized == True:
            self.qweight = self.unpack(
                self.qweight, self.qzeros.shape[1] * self.group_size
            )

            if self.device == "aie":
                self.qweight = self.qweight.transpose(0, 1)
                self.qzeros = self.qzeros.transpose(0, 1)
                self.scales = self.scales.to(torch.float).transpose(0, 1)
                AIEGEMM.select_op_handle()
                self.aiegemm = AIEGEMM.single_aiegemm
                self.wts_index = QLinearPerGrp.wts_cnt
                nodes = self.aiegemm.initialize_params(
                    self.qweight,
                    self.qzeros,
                    self.scales,
                    self.bias,
                    self.group_size,
                    dict(),
                    self.wts_index,
                )
                QLinearPerGrp.wts_cnt += nodes

                if not os.path.exists("./logs"):
                    os.makedirs("./logs")
                self._wire_npu()

            else:  # cpu
                self.weight = self.qweight - torch.repeat_interleave(
                    self.qzeros, self.group_size, dim=1
                )
                self.weight = self.weight * torch.repeat_interleave(
                    self.scales, self.group_size, dim=1
                )
                self.weight = self.weight.transpose(0, 1).to(torch.bfloat16)
                self.qzeros.to(torch.int8)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(torch.bfloat16)
                self.forward_func = self.forward_cpu
                self.qweight = None
                self.qzeros = None
                self.scales = None
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
        self.aiegemm.execute_aie(x, self.c_token, self.wts_index)
        return self.c_token

    def forward_aie_prefill_mladf(self, x: Tensor) -> Tensor:
        c = torch.zeros((x.shape[0], self.out_features), dtype=torch.bfloat16)
        self.aiegemm.execute_aie(x, c, self.wts_index)
        return c

    def forward_aie_prefill_mladf_bo_manager(self, x: Tensor) -> Tensor:
        out = self.aiegemm.execute_aie_bo_manager(x, self.wts_index)
        return out

    def forward_aie(self, x: Tensor) -> Tensor:
        return self.forward_dict_aie[x.shape[0]](x)

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

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
