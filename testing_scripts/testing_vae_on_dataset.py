import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm

import sys
import os
# Pretend we are running from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.dataset_configuration import get_trans10k_train_loader

"""
This script is used to test the VAE on the Trans10k dataset.
It runs both the images and masks through a VAE encode and decode cycle 
to determine what infomation is is preserved in the latent space for
the tranformer to use in denoising.
"""

# ==== Script Parameters ====
dataset_path = '/home/xycoord/models/Trans10k/'
train_batch_size = 1
device = "cuda"
dtype = torch.bfloat16
# Base Model that the VAE belongs to
# base_model_path = "stabilityai/stable-diffusion-3.5-large"
base_model_path = "stabilityai/stable-diffusion-2"


# ==== Data Loader ====
train_loader = get_trans10k_train_loader(
                dataset_path, 
                batch_size=train_batch_size, # Batch given Per GPU
               )

# ==== Load VAE ====
print("Loading VAE")
vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", use_safetensors=True, torch_dtype=dtype).to(device).requires_grad_(False).eval()
print("Done Loading VAE")

vae_image_processor = VaeImageProcessor(do_normalize=True)

# ==== Metrics ====
def contrast_score(image):
    image_shifted = image - 0.5
    return torch.mean(0.5-torch.abs(image_shifted))

def intersection_over_union(tensor1, tensor2):
    intersection = torch.sum((tensor1 * tensor2) > 0)
    union = torch.sum((tensor1 + tensor2) > 0)
    iou = intersection / union
    return iou


def quantize_tensor(tensor):
    return (tensor > 0.5).float()


total_iou = 0.0
batches = len(train_loader)

print("Starting Evaluation")

for step, batch in enumerate(tqdm(train_loader)):

    image, mask, name = batch

    # ==== Reshape data for VAE ====
    image = image.to(dtype)
    # mask is only a single channel so copy it across 3
    mask_stacked = mask.unsqueeze(1).repeat(1,3,1,1).to(dtype)
    # Map [0, 1] -> [-1, 1]
    image_normalized = vae_image_processor.normalize(image)
    mask_normalized= vae_image_processor.normalize(mask_stacked)

    # ==== Encode and Decode ====
    image_latents = vae.encode(image_normalized.to(device)).latent_dist.sample()
    image_output = vae.decode(image_latents).sample
    mask_latents = vae.encode(mask_normalized.to(device)).latent_dist.sample()
    mask_output = vae.decode(mask_latents).sample
    
    # ==== Postprocess ====
    image_output = image_output.cpu()
    mask_output = mask_output.cpu()
    # Clip and Map [-1, 1] -> [0, 1]
    mask_output_tensor = vae_image_processor.postprocess(mask_output, output_type='pt')
    # Quantize to 0 or 1
    mask_output_quantized = quantize_tensor(mask_output_tensor)

    image_output_pil = vae_image_processor.postprocess(image_output, output_type='pil')
    # Display the image output for visual inspection
    image_output_pil[0].show()

    # Compute Metrics
    iou = intersection_over_union(mask_stacked, mask_output_quantized)
    
    total_iou += iou


print("Average IOU: ", total_iou/batches)