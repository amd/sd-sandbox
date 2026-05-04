# Modifications Copyright (C) 2025 Advanced Micro Devices, 
# Inc.  All rights reserved.
#
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kolors.tokenization_chatglm import ChatGLMTokenizer
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers import XLMRobertaModel, ChineseCLIPTextModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
try:
    from diffusers.utils import randn_tensor
except:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.onnx_utils import OnnxRuntimeModel, ORT_TO_NP_TYPE

import warnings
import numpy as np
import PIL.Image
import torch.nn.functional as F
import time



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionKolorsPipelineAMD.from_pretrained(
        ...     "Kwai-Kolors/Kolors", torch_dtype=torch.float16
        ... )

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class StableDiffusionKolorsPipelineAMD(DiffusionPipeline, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Kolors with ONNX Runtime on AMD hardware.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae_encoder ([`OnnxRuntimeModel`]):
            ONNX Runtime model for VAE encoder to encode images to latent representations.
        vae_decoder ([`OnnxRuntimeModel`]):
            ONNX Runtime model for VAE decoder to decode latent representations to images.
        text_encoder ([`OnnxRuntimeModel`]):
            ONNX Runtime model for text encoder (ChatGLM) to encode text prompts.
            WARNING: Current model outputs logits [vocab_size=65024] instead of hidden_states [hidden_size=4096].
            This dimension mismatch may cause failures in subsequent pipeline stages.

        tokenizer (`ChatGLMTokenizer`):
            Tokenizer for ChatGLM model.

        unet ([`OnnxRuntimeModel`]): 
            ONNX Runtime model for conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        controlnet ([`OnnxRuntimeModel`], *optional*):
            Provides additional conditioning to the `unet` during the denoising process. If set, the ControlNet
            will be applied to guide the image generation process.
    """

    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,
        vae_decoder: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: ChatGLMTokenizer,
        unet: OnnxRuntimeModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        controlnet: OnnxRuntimeModel = None,
    ):
        super().__init__()

        self.register_modules(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(vae_decoder.config["block_out_channels"]) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.default_sample_size = self.unet.config["sample_size"]

        # self.watermark = StableDiffusionXLWatermarker()
        
        self.perf_time_dict = {
            "vae_encoder": [],
            "vae_decoder": [],
            "unet": [],
        }
        
        self.perf_time_gpu_model = {
            "tokenizer": [],
            "text_encoder": [],
            "tokenizer_negative": [],
            "text_encoder_negative": [],
        }

    def encode_prompt(
        self,
        prompt,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # from IPython import embed; embed(); exit()
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer]
        text_encoders = [self.text_encoder]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                
                # ONNX Runtime inference
                batch_size = text_inputs['input_ids'].shape[0]
                text_encoder_inputs = {
                    'input_ids': text_inputs['input_ids'].numpy().astype(np.int64),
                    'attention_mask': text_inputs['attention_mask'].numpy().astype(np.int64),
                    'position_ids': text_inputs['position_ids'].numpy().astype(np.int64),
                }
                
                # Check if text_encoder is ONNX or PyTorch model
                is_onnx_model = hasattr(text_encoder, 'model') and hasattr(text_encoder.model, 'run')
                
                t0 = time.time_ns()
                
                if is_onnx_model:
                    outputs = text_encoder.model.run(None, text_encoder_inputs)
                    prompt_embeds_np = outputs[0]  # [batch, seq_len, 4096] float16
                    pooled_prompt_embeds_np = outputs[0][:, -1, :]  # [batch, 4096] float16
                    
                    # Convert to torch tensors and permute; convert to float32 (UNet expects)
                    prompt_embeds = torch.from_numpy(prompt_embeds_np).float().permute(1, 0, 2)
                    pooled_prompt_embeds = torch.from_numpy(pooled_prompt_embeds_np).float()
                else:
                    # PyTorch inference (recommended for Kolors)
                    output = text_encoder(
                        input_ids=text_inputs['input_ids'].to(text_encoder.device),
                        attention_mask=text_inputs['attention_mask'].to(text_encoder.device),
                        position_ids=text_inputs['position_ids'].to(text_encoder.device),
                        output_hidden_states=True
                    )
                    # Extract embeddings from hidden_states
                    prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
                    pooled_prompt_embeds = output.hidden_states[-1][-1, :, :].clone()
                
                self.perf_time_gpu_model["text_encoder"].append((time.time_ns() - t0) * 1e-9)
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            # prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            prompt_embeds = prompt_embeds_list[0]

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            # negative_prompt = negative_prompt or ""
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                # ONNX Runtime inference for negative prompt
                neg_batch_size = uncond_input['input_ids'].shape[0]
                negative_text_inputs = {
                    'input_ids': uncond_input['input_ids'].numpy().astype(np.int64),
                    'attention_mask': uncond_input['attention_mask'].numpy().astype(np.int64),
                    'position_ids': uncond_input['position_ids'].numpy().astype(np.int64),
                }
                
                # Encode negative prompt (use same logic as positive prompt)
                is_onnx_model = hasattr(text_encoder, 'model') and hasattr(text_encoder.model, 'run')
                
                t0 = time.time_ns()
                
                if is_onnx_model:
                    outputs = text_encoder.model.run(None, negative_text_inputs)
                    negative_prompt_embeds_np = outputs[0]  # [batch, seq, 4096] float16
                    negative_pooled_prompt_embeds_np = outputs[0][:, -1, :]  # [batch, 4096] float16
                    
                    # Convert to torch tensors; convert to float32 (UNet expects)
                    negative_prompt_embeds = torch.from_numpy(negative_prompt_embeds_np).float().permute(1, 0, 2)
                    negative_pooled_prompt_embeds = torch.from_numpy(negative_pooled_prompt_embeds_np).float()
                else:
                    # PyTorch inference (recommended)
                    output = text_encoder(
                        input_ids=uncond_input['input_ids'].to(text_encoder.device),
                        attention_mask=uncond_input['attention_mask'].to(text_encoder.device),
                        position_ids=uncond_input['position_ids'].to(text_encoder.device),
                        output_hidden_states=True
                    )
                    negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
                    negative_pooled_prompt_embeds = output.hidden_states[-1][-1, :, :].clone()
                
                self.perf_time_gpu_model["text_encoder_negative"].append((time.time_ns() - t0) * 1e-9)

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype)

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_images_per_prompt, seq_len, -1
                    )

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            # negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
            negative_prompt_embeds = negative_prompt_embeds_list[0]

        bs_embed = pooled_prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        control_image=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        strength=None,
        num_inference_steps=None,
    ):
        if strength is not None and (strength < 0 or strength > 1):
            raise ValueError(f"The value of strength should be in [0.0, 1.0] but is {strength}")
        if num_inference_steps is not None:
            if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
                raise ValueError(
                    f"`num_inference_steps` has to be a positive integer but is {num_inference_steps} of type"
                    f" {type(num_inference_steps)}."
                )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.get("addition_time_embed_dim", 256) * len(add_time_ids) + 4096
        )
        # For ONNX models, we skip the validation check as we don't have direct access to model layers
        # expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        # if expected_add_embed_dim != passed_add_embed_dim:
        #     raise ValueError(
        #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        #     )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def _check_controlnet_inputs(self):
        """Check if UNet model supports ControlNet inputs."""
        if self.controlnet is None:
            return False
        
        unet_input_names = [inp.name for inp in self.unet.model.get_inputs()]
        # Check for typical ControlNet input names
        has_controlnet_inputs = any(
            "down_block" in name or "mid_block" in name or "additional_residual" in name
            for name in unet_input_names
        )
        return has_controlnet_inputs


    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width)
        if isinstance(image, list):
            image = torch.cat(image, dim=0)
        image = image.to(dtype=torch.float32)
        image_batch_size = image.shape[0]
        repeat_by = batch_size if image_batch_size == 1 else num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)
        return image

    def get_timesteps(self, num_inference_steps, strength, mode=None):
        # get the original timestep using init_timestep
        if mode == "controlnet_img2img" or mode == "inpainting":
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)
            return timesteps, num_inference_steps - t_start
        else:
            timesteps = self.scheduler.timesteps
            return timesteps, num_inference_steps


    def _clear_time_dict(self):
        """Clear performance timing dictionaries."""
        for key in self.perf_time_dict:
            self.perf_time_dict[key] = []
        
        for key in self.perf_time_gpu_model:
            self.perf_time_gpu_model[key] = []

    def _detect_mode(self, mask_image, control_image):
        """
        Auto-detect run mode (image and control_image merged into control_image).

        Returns:
            str: "text2img" | "inpainting" | "controlnet_img2img"
        """
        if mask_image is not None:
            return "inpainting"
        elif control_image is not None:
            return "controlnet_img2img"
        else:
            return "text2img"
    
    def prepare_image_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
        height=None,
        width=None,
    ):
        """
        Encode input image to latents and add noise (for img2img).

        Args:
            image: Input image (PIL.Image or torch.Tensor)
            timestep: Current timestep
            batch_size: Batch size
            num_images_per_prompt: Number of images per prompt
            dtype: Data type
            device: Device
            generator: Random number generator
            height: Target height for preprocess (align with reference)
            width: Target width for preprocess (align with reference)

        Returns:
            torch.FloatTensor: Noisy latents
        """
        import PIL.Image
        
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        
        # Preprocess image (with height/width when provided, align with reference)
        preprocess_kwargs = {}
        if height is not None and width is not None:
            preprocess_kwargs["height"] = height
            preprocess_kwargs["width"] = width
        image = self.image_processor.preprocess(image, **preprocess_kwargs)
        
        if isinstance(image, list):
            image = torch.cat(image, dim=0)
        
        if image.shape[1] == 4:
            # If already latent
            init_latents = image
        else:
            # Encode with VAE encoder
            if isinstance(self.vae_encoder, OnnxRuntimeModel):
                # ONNX version
                image_np = image.cpu().numpy().astype(np.float32)
                latents_np = self.vae_encoder(init_image=image_np)[0]
                init_latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
            else:
                # PyTorch version (fallback)
                init_latents = self.vae_encoder.encode(image.to(device=device, dtype=dtype)).latent_dist.sample(generator)
            
            # Apply scaling factor (Kolors uses 0.13025)
            scaling_factor = self.vae_encoder.config.get("scaling_factor", 0.18215)
            init_latents = init_latents * scaling_factor
        
        # Expand latents
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            init_latents = torch.cat([init_latents] * (batch_size // init_latents.shape[0]), dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)
        
        shape = init_latents.shape
        batch_size_actual = shape[0]
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if isinstance(timestep, torch.Tensor):
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).expand(batch_size_actual).to(device=device)
            elif timestep.numel() == 1 or timestep.shape[0] != batch_size_actual:
                timestep = timestep.flatten()[0:1].expand(batch_size_actual).to(device=device)
        else:
            timestep = torch.full((batch_size_actual,), timestep, device=device, dtype=torch.long)
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        
        return init_latents


    def prepare_mask_and_masked_image(
        self,
        image,
        mask,
        height,
        width,
        device,
        dtype,
        generator=None,
    ):
        """
        Prepare mask and masked image latents for inpainting.

        Args:
            image: Input image (PIL, tensor, or list)
            mask: Mask image (white = 1 = region to inpaint). PIL, tensor [0,1], or numpy.
            height: Target height
            width: Target width
            device: Device
            dtype: Data type
            generator: Random number generator

        Returns:
            tuple: (mask [B,1,H//8,W//8], masked_image_latents [B,4,H//8,W//8])
        """
        if image is None:
            raise ValueError("`image` input cannot be undefined.")
        if mask is None:
            raise ValueError("`mask` input cannot be undefined.")

        # Preprocess image (same as Kolors: normalize to [-1,1], BCHW)
        if isinstance(image, PIL.Image.Image):
            image = self.image_processor.preprocess(image, height=height, width=width)
        if isinstance(image, list):
            image = torch.cat(image, dim=0)
        image = image.to(dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Preprocess mask: [0,1], 1 = to inpaint (align with Kolors binarize mask >= 0.5 -> 1)
        if isinstance(mask, PIL.Image.Image):
            mask = mask.convert("L")
            mask = mask.resize((width, height), PIL.Image.Resampling.LANCZOS)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        elif isinstance(mask, torch.Tensor):
            mask = mask.to(dtype=dtype, device=device)
        elif isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(dtype=dtype, device=device)
        
        # Ensure mask is 4D tensor [batch, 1, height, width]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] != image.shape[0]:
            mask = mask.repeat(image.shape[0], 1, 1, 1)
        # Binarize: 1 = to inpaint (Kolors convention)
        mask = (mask >= 0.5).to(dtype=torch.float32)

        # masked_image = image where we do NOT inpaint (keep known pixels)
        masked_image = image * (1.0 - mask)

        # Latent size (8x for SD/Kolors)
        latent_h, latent_w = height // self.vae_scale_factor, width // self.vae_scale_factor

        vae_encoder = getattr(self, "vae_encoder", None)
        scaling_factor = getattr(vae_encoder.config, "scaling_factor", None) or 0.18215
        if vae_encoder is not None:
            if isinstance(vae_encoder, OnnxRuntimeModel):
                masked_image_np = masked_image.cpu().numpy().astype(np.float32)
                masked_latents_np = vae_encoder(init_image=masked_image_np)[0]
                masked_image_latents = torch.from_numpy(masked_latents_np).to(device=device, dtype=dtype)
            else:
                masked_image_latents = vae_encoder.encode(
                    masked_image.to(device=device, dtype=dtype)
                ).latent_dist.sample(generator)
            scaling_factor = getattr(vae_encoder.config, "scaling_factor", None) or 0.18215
            masked_image_latents = masked_image_latents * scaling_factor
        else:
            # No vae_encoder: use zero latents so UNet still gets 9 channels (degraded inpainting)
            batch_size = masked_image.shape[0]
            masked_image_latents = torch.zeros(
                batch_size, 4, latent_h, latent_w, device=device, dtype=dtype
            )
            warnings.warn(
                "Inpainting without vae_encoder: masked_image_latents are zeros; quality may be degraded.",
                UserWarning,
                stacklevel=2,
            )

        # Downsample mask to latent spatial size (same as Kolors prepare_mask_latents)
        mask = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]]] = None,
        strength: float = 0.8,
        # Inpainting parameters
        mask_image = None,
        # Controlnet patameters
        control_image = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        # Common parameters
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        use_dynamic_threshold: Optional[bool] = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            control_image (`PIL.Image.Image`, `torch.FloatTensor`, `List[PIL.Image.Image]`, `List[torch.FloatTensor]`, *optional*):
                Single image used for both ControlNet conditioning and img2img initial image. When provided, it is
                encoded to latent space and used to guide the generation and (with `strength`) as the init image.
                For inpainting, this is the image to inpaint (together with `mask_image`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. For instance, if denoising_end is set to
                0.7 and `num_inference_steps` is fixed at 50, the process will execute only 35 (i.e., 0.7 * 50)
                Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 7.5):
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] instead of a
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                Original size of the image before cropping or resizing.
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                Top-left coordinates of the crop if the image was cropped.
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                Target size of the generated image.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The strength of the ControlNet conditioning. Set to 1.0 for full strength. Can be a list for multiple
                ControlNets.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying (0.0 = start, 1.0 = end).
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying (0.0 = start, 1.0 = end).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # Clear performance timing dictionaries
        self._clear_time_dict()
        
        # 0. Detect pipeline mode
        mode = self._detect_mode(mask_image, control_image)
        
        # 0.0. For img2img: image = init image, control_image = ControlNet only; when image is None use control_image for both (backward compat)
        init_image = image if image is not None else control_image
        
        # 0.1. For img2img/inpainting, default height/width from control_image when not provided
        if mode in ["controlnet_img2img", "inpainting"] and init_image is not None:
            if isinstance(init_image, PIL.Image.Image):
                height = height or init_image.size[1]
                width = width or init_image.size[0]
            elif isinstance(init_image, torch.Tensor):
                height = height or init_image.shape[2]
                width = width or init_image.shape[3]
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            control_image,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            strength=strength,
            num_inference_steps=num_inference_steps,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, mode)

        # 5. Prepare latent variables
        num_channels_latents = self.vae_decoder.config.get("latent_channels", 4)

        # Prepare ControlNet conditioning if control_image is provided
        controlnet_cond = None
        if control_image is not None and self.controlnet is not None:
            controlnet_cond = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

        # 5.1. Prepare latents based on mode
        mask = None
        masked_image_latents = None
        
        if mode == "controlnet_img2img":
            # Img2img: strength>=1 use random latents, else encode init_image and add noise (align with reference)
            if strength >= 1.0:
                latents = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )
            else:
                latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
                latents = self.prepare_image_latents(
                    init_image,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    height=height,
                    width=width,
                )
        elif mode == "inpainting":
            mask, masked_image_latents = self.prepare_mask_and_masked_image(
                control_image,
                mask_image,
                height,
                width,
                device,
                prompt_embeds.dtype,
                generator,
            )
            # strength>=1 pure noise, strength<1 image latents + noise
            if strength >= 1.0:
                latents = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )
            else:
                latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
                latents = self.prepare_image_latents(
                    control_image,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    height=height,
                    width=width,
                )
        else:
            # Text2img / ControlNet: prepare random latents
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 7.5. Prepare ControlNet conditioning schedule
        controlnet_keep = None
        if self.controlnet is not None and controlnet_cond is not None:
            # Create controlnet_keep list to control when ControlNet is applied
            controlnet_keep = []
            for step_idx in range(len(timesteps)):
                # Calculate if ControlNet should be applied at this step
                step_ratio = step_idx / len(timesteps)
                next_step_ratio = (step_idx + 1) / len(timesteps)
                
                # Convert to lists if single values
                start_list = [control_guidance_start] if isinstance(control_guidance_start, float) else control_guidance_start
                end_list = [control_guidance_end] if isinstance(control_guidance_end, float) else control_guidance_end
                
                keeps = [
                    1.0 - float(step_ratio < s or next_step_ratio > e)
                    for s, e in zip(start_list, end_list)
                ]
                controlnet_keep.append(keeps[0])

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if denoising_end is not None:
            num_inference_steps = int(round(denoising_end * num_inference_steps))
            timesteps = timesteps[: num_warmup_steps + self.scheduler.order * num_inference_steps]

        def _get_model_input_dtype(model, input_name, fallback="tensor(float)"):
            input_type = next(
                (inp.type for inp in model.get_inputs() if inp.name == input_name),
                fallback,
            )
            return ORT_TO_NP_TYPE[input_type]

        unet_sample_dtype = _get_model_input_dtype(self.unet.model, "sample")
        unet_encoder_hidden_states_dtype = _get_model_input_dtype(
            self.unet.model, "encoder_hidden_states"
        )
        unet_text_embeds_dtype = _get_model_input_dtype(self.unet.model, "text_embeds")

        controlnet_sample_dtype = None
        controlnet_encoder_hidden_states_dtype = None
        controlnet_text_embeds_dtype = None
        controlnet_controlnet_cond_dtype = None
        controlnet_conditioning_scale_dtype = None
        if self.controlnet is not None:
            controlnet_sample_dtype = _get_model_input_dtype(self.controlnet.model, "sample")
            controlnet_encoder_hidden_states_dtype = _get_model_input_dtype(
                self.controlnet.model, "encoder_hidden_states"
            )
            controlnet_text_embeds_dtype = _get_model_input_dtype(
                self.controlnet.model, "text_embeds"
            )
            controlnet_controlnet_cond_dtype = _get_model_input_dtype(
                self.controlnet.model, "controlnet_cond"
            )
            controlnet_conditioning_scale_dtype = _get_model_input_dtype(
                self.controlnet.model, "conditioning_scale"
            )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Inpainting: concatenate mask and masked_image_latents
                if mode == "inpainting" and mask is not None and masked_image_latents is not None:
                    # Expand mask and masked_image_latents for CFG
                    mask_input = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
                    masked_image_input = torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
                    # Concatenate: [batch, 4+1+4, h, w] = [batch, 9, h, w]
                    latent_model_input = torch.cat([latent_model_input, mask_input, masked_image_input], dim=1)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                
                # Prepare timestep for ONNX model
                timestep_dtype = next(
                    (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"),
                    "tensor(float)",
                )
                timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
                timestep = (
                    np.ones(self.unet.model.get_inputs()[1].shape, dtype=timestep_dtype)
                    * t.item()
                ).astype(timestep_dtype)
                
                # Get add_time_ids dtype
                add_ids_dtype = next(
                    (input.type for input in self.unet.model.get_inputs() if input.name == "time_ids"),
                    "tensor(float)",
                )
                add_ids_dtype = ORT_TO_NP_TYPE[add_ids_dtype]
                
                # Prepare base UNet inputs
                unet_inputs = {
                    "sample": (
                        latent_model_input.cpu().numpy().astype(unet_sample_dtype)
                        if isinstance(latent_model_input, torch.Tensor)
                        else latent_model_input.astype(unet_sample_dtype)
                    ),
                    "timestep": timestep,
                    "encoder_hidden_states": (
                        prompt_embeds.cpu().numpy().astype(unet_encoder_hidden_states_dtype)
                        if isinstance(prompt_embeds, torch.Tensor)
                        else prompt_embeds.astype(unet_encoder_hidden_states_dtype)
                    ),
                    "text_embeds": (
                        add_text_embeds.cpu().numpy().astype(unet_text_embeds_dtype)
                        if isinstance(add_text_embeds, torch.Tensor)
                        else add_text_embeds.astype(unet_text_embeds_dtype)
                    ),
                    "time_ids": add_time_ids.numpy().astype(add_ids_dtype),
                }
                
                # Apply ControlNet if available
                if self.controlnet is not None and controlnet_cond is not None and controlnet_keep is not None:
                    # Calculate conditioning scale for this step
                    if isinstance(controlnet_conditioning_scale, list):
                        cond_scale = controlnet_conditioning_scale[0]
                    else:
                        cond_scale = controlnet_conditioning_scale
                    cond_scale = cond_scale * controlnet_keep[i]
                    
                    # Run ControlNet to get control features
                    controlnet_inputs = {
                        "sample": unet_inputs["sample"].astype(controlnet_sample_dtype),
                        "timestep": timestep,
                        "encoder_hidden_states": unet_inputs["encoder_hidden_states"].astype(
                            controlnet_encoder_hidden_states_dtype
                        ),
                        "controlnet_cond": (
                            controlnet_cond.cpu().numpy().astype(controlnet_controlnet_cond_dtype)
                            if isinstance(controlnet_cond, torch.Tensor)
                            else controlnet_cond.astype(controlnet_controlnet_cond_dtype)
                        ),
                        "conditioning_scale": np.array(
                            [cond_scale], dtype=controlnet_conditioning_scale_dtype
                        ),
                        "text_embeds": unet_inputs["text_embeds"].astype(
                            controlnet_text_embeds_dtype
                        ),
                        "time_ids": unet_inputs["time_ids"],
                    }
                    
                    # Get control block samples from ControlNet
                    controlnet_outputs = self.controlnet(**controlnet_inputs)
                    
                    # Add control features to UNet inputs
                    # ControlNet typically returns down_block_res_samples and mid_block_res_sample
                    if len(controlnet_outputs) >= 2:
                        # Multiple down blocks + mid block
                        for idx, down_block_sample in enumerate(controlnet_outputs[:-1]):
                            unet_inputs[f"down_block_res_sample_{idx}"] = down_block_sample
                        unet_inputs["mid_block_res_sample"] = controlnet_outputs[-1]
                
                start = time.time_ns()
                noise_pred = self.unet(**unet_inputs)[0]
                end = time.time_ns()
                noise_pred = torch.tensor(noise_pred)
                self.perf_time_dict["unet"].append((end - start) * 1e-9)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if use_dynamic_threshold:
                        DynamicThresh = DynThresh(maxSteps=num_inference_steps, experiment_mode=0)
                        noise_pred = DynamicThresh.dynthresh(noise_pred_text,
                            noise_pred_uncond,
                            guidance_scale,
                            None)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Decode latents with ONNX VAE decoder
        if not output_type == "latent":
            # Get scaling factor from config
            scaling_factor = self.vae_decoder.config.get("scaling_factor", 0.18215)
            latents = latents / scaling_factor
            latents = latents.cpu().numpy()
            
            start = time.time_ns()
            # Batch decode if needed
            if latents.shape[0] > 1:
                image = np.concatenate(
                    [
                        self.vae_decoder(
                            **{self.vae_decoder.model.get_inputs()[0].name: latents[i : i + 1]}
                        )[0]
                        for i in range(latents.shape[0])
                    ]
                )
            else:
                image = self.vae_decoder(**{self.vae_decoder.model.get_inputs()[0].name: latents})[0]
            image = torch.tensor(image)
            end = time.time_ns()
            self.perf_time_dict["vae_decoder"].append((end - start) * 1e-9)
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # image = self.watermark.apply_watermark(image)
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


if __name__ == "__main__":
    pass