#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.
#

from safetensors.torch import load_file
from safetensors import safe_open
import torch
import argparse
import torch._C._onnx as _C_onnx

from diffusers import AutoencoderKL
from diffusers import ControlNetModel
from diffusers import UNet2DConditionModel

from onnxconverter_common import float16
from pathlib import Path
import onnx
import os


def is_fp16_safetensors(safetensor_path):
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        return all(f.get_tensor(key).dtype == torch.float16 for key in f.keys())


def convert_fp16_onnx_model(
    onnx_model, dst_onnx_path, save_as_external_data=True, location=""
):
    return
    onnx_fp16_model = float16.convert_float_to_float16(
        onnx_model, disable_shape_infer=True
    )
    if save_as_external_data:
        if location == "":
            location = Path(dst_onnx_path).name + "_data"
        if os.path.exists(Path(dst_onnx_path).parent / location):
            Path.unlink(Path(dst_onnx_path).parent / location)
    onnx.save_model(
        onnx_fp16_model,
        dst_onnx_path,
        save_as_external_data=save_as_external_data,
        all_tensors_to_one_file=True,
        location=location,
    )


def reorganize_onnx_external_data(old_onnx_path, new_onnx_path, location):
    model = onnx.load(old_onnx_path, load_external_data=False)
    graph = model.graph
    unlink_list = []
    for init in graph.initializer:
        if onnx.external_data_helper.uses_external_data(init):
            for ext in init.external_data:
                if ext.key == "location":
                    unlink_list.append(ext.value)
    model = onnx.load(old_onnx_path)
    if os.path.exists(Path(new_onnx_path).parent / location):
        Path.unlink(Path(new_onnx_path).parent / location)
    onnx.save(
        model,
        new_onnx_path,
        save_as_external_data=True,
        location=location,
    )
    for f in unlink_list:
        Path.unlink(Path(old_onnx_path).parent / Path(f))
    if new_onnx_path != old_onnx_path:
        Path.unlink(Path(old_onnx_path))


def export_vae_decoder(safetensor_path, config_path, onnx_path):
    vae = AutoencoderKL.from_single_file(safetensor_path)
    vae.eval()

    vae_decoder = vae.decoder
    dummy_input = torch.randn(1, 4, 64, 64)  # (batch_size, latent_dim, height, width)

    torch.onnx.export(
        vae_decoder,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        onnx_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["sample"],  # the model's input names
        output_names=["image"],  # the model's output names
    )

    # if is_fp16_safetensors(safetensor_path):
    #     convert_fp16_onnx_model(onnx.load(onnx_path), onnx_path, False)

    print(f"VAE Decoder exported to {onnx_path}")


def export_controlnet(safetensor_path, config_path, onnx_path):
    config = ControlNetModel.load_config(config_path)
    model = ControlNetModel.from_config(config)

    state_dict = load_file(safetensor_path)
    model.load_state_dict(state_dict, strict=True)

    batch_size = 2
    seq_length = 77
    hidden_dim = 768
    image_shape = (
        batch_size,
        3,
        512,
        512,
    )
    latent_shape = (
        batch_size,
        4,
        64,
        64,
    )
    sample = torch.randn(latent_shape)
    timestep = torch.tensor([1] * batch_size, dtype=torch.int64)
    encoder_hidden_states = torch.randn(batch_size, seq_length, hidden_dim)
    controlnet_cond = torch.randn(image_shape)
    conditioning_scale = 1.0

    inputs = {
        "sample": sample,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "controlnet_cond": controlnet_cond,
        "conditioning_scale": conditioning_scale,
        # "class_labels": class_labels,
        # "timestep_cond": timestep_cond,
        # "attention_mask": attention_mask,
        # "added_cond_kwargs": added_cond_kwargs,
        # "cross_attention_kwargs": cross_attention_kwargs,
        # "guess_mode": guess_mode,
        # "return_dict": return_dict,
    }

    model_input = (
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale,
    )

    torch.onnx.export(
        model,
        model_input,
        onnx_path,
        input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            "controlnet_cond",
            "conditioning_scale",
        ],
        output_names=[
            "down_block_res_samples",
            "mid_block_res_sample",
            "control_block_samples_1",
            "control_block_samples_2",
            "control_block_samples_3",
            "control_block_samples_4",
            "control_block_samples_5",
            "control_block_samples_6",
            "control_block_samples_7",
            "control_block_samples_8",
            "control_block_samples_9",
            "control_block_samples_10",
            "control_block_samples_11",
        ],
        opset_version=17,
        # dynamic_axes={
        #     "sample": {0: "batch", 2: "height", 3: "width"},
        #     "controlnet_cond": {0: "batch", 2: "image_h", 3: "image_w"},
        #     "encoder_hidden_states": {0: "batch", 1: "seq"},
        #     "timestep": {0: "batch"},
        # },
        operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        external_data_format=True,
    )

    # if is_fp16_safetensors(safetensor_path):
    #     convert_fp16_onnx_model(onnx.load(onnx_path), onnx_path, False)


def export_unet(safetensor_path, config_path, onnx_path, with_controlnet=True):
    config = UNet2DConditionModel.load_config(config_path)
    model = UNet2DConditionModel.from_config(config)
    state_dict = load_file(safetensor_path)
    model.load_state_dict(state_dict, strict=True)

    batch_size, channels, height, width = 2, 4, 64, 64
    sample = torch.randn(batch_size, channels, height, width)
    timestep = torch.tensor([1] * batch_size, dtype=torch.int64)

    seq_length = 77
    hidden_dim = 768
    encoder_hidden_states = torch.randn(batch_size, seq_length, hidden_dim)

    class_labels = None
    timestep_cond = None
    attention_mask = None

    if with_controlnet:
        down_block_additional_residuals = (
            torch.randn(batch_size, 320, height, width),
            torch.randn(batch_size, 320, height, width),
            torch.randn(batch_size, 320, height, width),
            torch.randn(batch_size, 320, height // 2, width // 2),
            torch.randn(batch_size, 640, height // 2, width // 2),
            torch.randn(batch_size, 640, height // 2, width // 2),
            torch.randn(batch_size, 640, height // 4, width // 4),
            torch.randn(batch_size, 1280, height // 4, height // 4),
            torch.randn(batch_size, 1280, height // 4, height // 4),
            torch.randn(batch_size, 1280, height // 8, height // 8),
            torch.randn(batch_size, 1280, height // 8, height // 8),
            torch.randn(batch_size, 1280, height // 8, height // 8),
        )
        mid_block_additional_residual = torch.randn(
            batch_size, 1280, height // 8, width // 8
        )
    else:
        down_block_additional_residuals = None
        mid_block_additional_residual = None

    model_input = (
        sample,
        timestep,
        encoder_hidden_states,
        None,
        None,
        None,
        None,
        None,
        down_block_additional_residuals,
        mid_block_additional_residual,
    )
    torch.onnx.export(
        model,
        model_input,
        onnx_path,
        input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            "class_labels",
            "timestep_cond",
            "attention_mask",
            "cross_attention_kwargs",
            "added_cond_kwargs",
            "down_block_additional_residuals",
            "mid_block_additional_residual",
        ],
        output_names=["noise_pred"],  # the model's output names
        opset_version=17,
        # dynamic_axes={
        #     "sample": {0: "batch", 2: "height", 3: "width"},
        #     "timestep": {0: "batch"},
        #     "encoder_hidden_states": {0: "batch", 1: "seq"},
        # },
        operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        external_data_format=True,
    )
    reorganize_onnx_external_data(onnx_path, onnx_path, Path(onnx_path).name + "_data")

    # if is_fp16_safetensors(safetensor_path):
    #     convert_fp16_onnx_model(onnx.load(onnx_path), onnx_path, True)


if __name__ == "__main__":
    export_unet(
        "./unet/diffusion_pytorch_model.fp16.safetensors",
        "./unet/config.json",
        "./unet/model.onnx",
    )
    export_controlnet(
        "./controlnet/diffusion_pytorch_model.fp16.safetensors",
        "./controlnet/config.json",
        "./controlnet/model.onnx",
    )
    export_vae_decoder(
        "./vae/diffusion_pytorch_model.fp16.safetensors",
        "./vae/config.json",
        "./vae/model.onnx",
    )
