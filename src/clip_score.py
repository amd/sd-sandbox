# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np
from PIL import Image
import argparse

# Partial function to specify the CLIP model to be used
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def load_and_prepare_image(image_path: str):
    """Load an image, convert it to a numpy array, and prepare it for CLIP scoring."""
    image = Image.open(image_path).convert("RGB")
    # Convert image to numpy array and ensure it has the correct type and scale
    image_np = np.array(image, dtype=np.uint8)
    # Add a batch dimension if necessary
    if image_np.ndim == 3:
        image_np = np.expand_dims(image_np, axis=0)
    return image_np


def calculate_clip_score(images, prompts, negative_prompts=None):
    """Calculate the CLIP score for a batch of images and a list of prompts."""
    images_int = (images * 255).astype("uint8")
    # Adjust dimensions to match [batch, channels, height, width])
    clip_scores = clip_score_fn(
        torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts
    ).detach()
    if negative_prompts:
        negative_score = clip_score_fn(
            torch.from_numpy(images_int).permute(0, 3, 1, 2), negative_prompts
        ).detach()
        return round(float(clip_scores.mean()), 4), round(
            float(negative_score.mean()), 4
        )
    return round(float(clip_scores.mean()), 4), None


def calc_clip_score(image, prompts, negative_prompts=None):
    image_np = np.array(image, dtype=np.uint8)
    # Add a batch dimension if necessary
    if image_np.ndim == 3:
        image_np = np.expand_dims(image_np, axis=0)
    pos_score, neg_score = calculate_clip_score(image_np, prompts, negative_prompts)
    # print(f"CLIP score: {sd_clip_score}")
    return pos_score, neg_score


def run(prompts, image_path):

    # Load and prepare the image
    images = load_and_prepare_image(image_path)

    # Calculate the CLIP score
    sd_clip_score = calculate_clip_score(images, prompts)
    print(f"CLIP score: {sd_clip_score}")


def check_config(args):
    if args.image == None:
        print(f" *** MODE NOT SUPPORTED *** : check help and readme")
        raise SystemExit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Input image path", type=str)
    args = parser.parse_args()

    check_config(args)

    prompts = [
        "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner"
    ]  # Replace with your text description

    run(prompts, args.image)
