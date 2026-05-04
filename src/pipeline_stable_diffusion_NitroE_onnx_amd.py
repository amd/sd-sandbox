# Modifications Copyright (C) 2025 Advanced Micro Devices, 
# Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch

import time
import numpy as np
import logging as sd3Logger
from diffusers.pipelines.onnx_utils import OnnxRuntimeModel, ORT_TO_NP_TYPE
from PIL import Image

from transformers import (
    LlamaForCausalLM,
)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`FlowMatchEulerDiscrete`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionNitroEONNXPipelineAMD(
    DiffusionPipeline,
):
    model_cpu_offload_seq = (
        "tokenizer->text_encoder->transformer->vae_decoder"
    )

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        tokenizer: LlamaForCausalLM,
        text_encoder: OnnxRuntimeModel = None,
        transformer: OnnxRuntimeModel = None,
        vae_decoder: OnnxRuntimeModel = None,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
            vae_decoder=vae_decoder,
        )

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.vae_decoder = vae_decoder

        self.perf_time_dict = {
            "vae_decoder": [],
            "transformer": [],
        }

        self.perf_time_gpu_model = {
            "tokenizer": [],
            "text_encoder": [],
        }

    def _clear_time_dict(
        self,
    ):
        for key in self.perf_time_dict:
            self.perf_time_dict[key] = []

        for key in self.perf_time_gpu_model:
            self.perf_time_gpu_model[key] = []

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        clean_caption: bool = True,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=max_sequence_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        text_encoder_inputs = {
            'input_ids':(text_input_ids.numpy()).astype('int64'), 
            'attention_mask':(prompt_attention_mask.numpy()).astype('int64'),
            'position_ids':(np.arange(max_sequence_length)[None,:]).astype('int64'), 
        }
        outputs = self.text_encoder.model.run(None,text_encoder_inputs)
        prompt_embeds = outputs[0]

        if self.do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size \
                if isinstance(negative_prompt, str) else negative_prompt
            uncond_input = self.tokenizer(
                            uncond_tokens,
                            padding="max_length",
                            max_length=max_sequence_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_text_inputs = {
                'input_ids':(uncond_input.input_ids.numpy()).astype('int64'),
                'attention_mask':(prompt_attention_mask.numpy()).astype('int64'),
                'position_ids':(np.arange(max_sequence_length)[None,:]).astype('int64'), 
            }
            outputs = self.text_encoder.model.run(None,negative_text_inputs) 
            negative_prompt_embeds = outputs[0]

        if not self.do_classifier_free_guidance:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
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
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

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

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}"
            )

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_decoder.config["scale_factor"],
            int(width) // self.vae_decoder.config["scale_factor"],
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def image_postprocess(self, images):
        pil_images = []
        for i in range(images.shape[0]):
            image = np.expand_dims(images[i,:], axis=0)
            image = np.clip(image * 0.5 + 0.5, 0, 1)
            image = np.transpose(image, (0,2,3,1)) # [N,C,H,W]--->[N,H,W,C]
            image = (image * 255).round().astype('uint8')
            # from array to pil
            pil_image = Image.fromarray(image[0])
            pil_images.append(pil_image)
        return pil_images

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale: float = 4.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
    ):
        sd3Logger.debug(f"max_sequence_length : {max_sequence_length}")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        device = torch.device("cpu")
        dtype = torch.float # torch.float16

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = np.concatenate(
                [negative_prompt_embeds,prompt_embeds], axis=0
            )
            prompt_attention_mask = np.concatenate(
                [negative_prompt_attention_mask, prompt_attention_mask], axis=0
            )
            prompt_embeds = torch.from_numpy(prompt_embeds).to(device=device, dtype=dtype)
            prompt_attention_mask = torch.from_numpy(prompt_attention_mask).to(device=device, dtype=dtype)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device,
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config["in_channels"]
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype).cpu().numpy()

        # 6. Denoising loop
        t0 = time.perf_counter()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                latent_model_input = (
                    latent_model_input.to(dtype=dtype).cpu().numpy()
                )
                timestep_dtype = next(
                    (
                        input.type
                        for input in self.transformer.model.get_inputs()
                        if input.name == "timestep"
                    ),
                    "tensor(float)",
                )
                timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
                timestep = timestep.to(dtype=dtype).cpu().numpy().astype(timestep_dtype)

                model_input = {
                    'hidden_states':latent_model_input,
                    'encoder_hidden_states':prompt_embeds,
                    'timestep':timestep, 
                }

                transformer_start_time = time.perf_counter()
                noise_pred = self.transformer.model.run(None, model_input)
                transformer_time = time.perf_counter() - transformer_start_time
                self.perf_time_dict["transformer"].append(transformer_time)
                noise_pred = torch.from_numpy(noise_pred[0]).to(device)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                # print("t: ", t, " | ", latents.min(), latents.max(), latents.mean(), latents.std())

                progress_bar.update()

        sd3Logger.debug(
            f"transformer inference total time ({num_inference_steps} steps) = {time.perf_counter() - t0:.3f}s"
        )

        t0 = time.perf_counter()
        latents = (
            latents / self.vae_decoder.config["scaling_factor"]
        ) + self.vae_decoder.config["shift_factor"]
        if latents.shape[0] > 1:
            image = np.concatenate(
                [
                    self.vae_decoder(latent_sample=latents[i : i + 1].cpu().numpy())[0]
                    for i in range(latents.shape[0])
                ]
            )
        else:
            image = self.vae_decoder(latent_sample=latents.cpu().numpy())[0]
        sd3Logger.debug(f"vae_decoder inference time = {time.perf_counter() - t0:.3f}s")
        self.perf_time_dict["vae_decoder"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        image = self.image_postprocess(image)
        sd3Logger.debug(
            f"image_processor postprocess time = {time.perf_counter() - t0:.4f}s"
        )

        # dump perf counters
        for k, v in self.perf_time_dict.items():
            sd3Logger.debug(f"==> {k} : exec time {len(v)}, avg time {sum(v)/len(v)}")

        return StableDiffusion3PipelineOutput(images=image)
