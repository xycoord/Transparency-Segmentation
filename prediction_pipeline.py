from PIL import Image
from typing import Union
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusers import DiffusionPipeline, AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory

from utils.utils import load_prompt_embeds


class MaskPipelineOutput(BaseOutput):
    mask_np: torch.Tensor
    mask_pil: Image.Image
    mask_vae: Image.Image
    uncertainty: Union[None, np.ndarray]


class PredictionPipeline(DiffusionPipeline):

    def __init__(self,
                 transformer:SD3Transformer2DModel,
                 vae:AutoencoderKL,
                 noise_scheduler:FlowMatchEulerDiscreteScheduler,
                 use_zeros_start:bool = False,
                 quantize:bool = True,
                 ):
        super().__init__()
            
        self.register_modules(
            transformer=transformer,
            vae=vae,
            noise_scheduler=noise_scheduler,
        )
        self.use_zeros_start = use_zeros_start
        self.quantize = quantize

        self.vae_image_processor = VaeImageProcessor(do_normalize=True)

        # ==== Load Precomputed Prompt Embeddings ====
        prompt_embeds_path = './precomputed_prompt_embeddings/'
        prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(prompt_embeds_path)
        self.prompt_embeds = prompt_embeds.to(self.device)
        self.pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)


    @torch.no_grad()
    def __call__(self,
                 input_images: torch.Tensor = None,
                 input_masks: torch.Tensor = None,
                 denoise_steps: int =10,
                 ensemble_size: int =10,
                 batch_size:int =1,
                 ) -> MaskPipelineOutput:
        
        """
        Process images and masks through the prediction pipeline to generate mask outputs.

        Args:
            input_images (torch.Tensor, optional): Input images to predict from. 
            input_masks (torch.Tensor, optional): Input masks to pass through the vae as a control.
            denoise_steps (int, optional): Number of denoising steps.
            ensemble_size (int, optional): Number of predictions to ensemble per image. 
            batch_size (int, optional): Determines how many desnoising processes to run in parallel.

        Returns:
            MaskPipelineOutput: A named tuple containing:
                - mask_pt (torch.Tensor): The predicted masks in PyTorch tensor format
                - mask_pil (PIL.Image): The colored mask visualization
                - mask_vae (PIL.Image): The VAE-processed mask
                - uncertainty (None): Placeholder for uncertainty metrics
        """

        # ==== Prepare Prompt Embeds ====
        prompt_embeds_batch = self.prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds_batch = self.pooled_prompt_embeds.repeat(batch_size, 1)


        # ==== Preprare for Encoder ====
        images_normalized = self.vae_image_processor.normalize(input_images).to(self.device).to(torch.bfloat16) # Shouldn't be hard coded type
        mask_normalized = self.vae_image_processor.normalize(input_masks).to(self.device).to(torch.bfloat16) # Shouldn't be hard coded type
        
        # ==== Encode ====
        image_latents = self.vae.encode(images_normalized).latent_dist.sample()
        mask_latents = self.vae.encode(mask_normalized).latent_dist.sample()

        # ==== Prepare for Transformer ====
        # Scale so latents behave well with diffusion
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        mask_latents = (mask_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        output = self.generate_ensembled_mask(ensemble_size, image_latents, denoise_steps, batch_size, prompt_embeds_batch, pooled_prompt_embeds_batch)

        # decode the output
        mask_latents = (mask_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        mask = self.vae.decode(mask_latents).sample
        
        mask_pred = output
        mask_pred_np = self.vae_image_processor.pt_to_numpy(mask_pred)
        output_PIL = self.vae_image_processor.numpy_to_pil(mask_pred_np)
        mask_pt = self.vae_image_processor.postprocess(mask, output_type='pt')
        mask_PIL = self.vae_image_processor.postprocess(mask, output_type='pil')


        return MaskPipelineOutput(mask_pt=mask_pred, mask_pil=output_PIL, mask_vae=mask_PIL, uncertainty=None)
    
    def generate_ensembled_mask(self, ensemble_size, image_latents, denoise_steps, batch_size, prompt_embeds_batch, pooled_prompt_embeds_batch):
        num_images = image_latents.shape[0]

        # To parelalise ensembling with an independent batch size, we create a dataloader with the image repeated
        # (num_images * ensemble_size, 16, 128, 128)
        # Say we have 3 images (a, b, c) and ensemble size 2, we get: (a, a, b, b, c, c)
        repeated_image_latents = image_latents.repeat_interleave(repeats=ensemble_size, dim=0)

        ensemble_image_dataset = TensorDataset(repeated_image_latents)
        batch_size = batch_size if batch_size > 0 else 1 
        ensemble_image_loader = DataLoader(ensemble_image_dataset,batch_size=batch_size,shuffle=False)

        mask_preds = []
        
        iterable_bar = tqdm(
            ensemble_image_loader, desc=" " * 2 + "Inference batches", leave=False
        )
        
        for batch in iterable_bar:
            (batched_image,)= batch  # here the image is still around 0-1
            self.noise_scheduler.set_timesteps(denoise_steps, device=self.device)
            output_latents, noise = self.generate_single_mask(batch_size, batched_image, prompt_embeds_batch, pooled_prompt_embeds_batch)
            output_latents = (output_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            mask_pred_raw = self.vae.decode(output_latents).sample
            mask_pred_raw = self.vae_image_processor.postprocess(mask_pred_raw, output_type='pt')
            mask_preds.append(mask_pred_raw.detach().clone())
            free_memory()

        raw_preds = torch.cat(mask_preds, dim=0)
        latent_shape = raw_preds.shape[1:]
        #  Unflatten the batch dimension so each image is independently ensembled
        raw_preds = raw_preds.reshape(num_images, ensemble_size, *latent_shape)

        ensembled_preds = self.ensemble(raw_preds)

        del raw_preds
        del mask_preds
        del ensemble_image_dataset
        del ensemble_image_loader
        free_memory()
        return ensembled_preds

    def ensemble(self, raw_preds):
        """
        Args:
            raw_preds (torch.Tensor): (num_images, ensemble_size, 3, width, height)
        return:
            torch.Tensor: (num_images, 3, width, height)
        """

        # Average over both the ensemble size and the channel dimension as a way of voting per pixel    
        channel_mean = raw_preds.mean(dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        ensemble_mean = channel_mean.mean(dim=1, keepdim=False)

        # Quantize each pixel to 0 or 1
        if self.quantize:
            quantized_mean = self.quantize_tensor(ensemble_mean)
            return quantized_mean 
        else:
            return ensemble_mean

    def quantize_tensor(self, tensor):
        return (tensor > 0.5).float()

    def generate_single_mask(self, batch_size, image_latents, prompt_embeds_batch, pooled_prompt_embeds_batch):
        # ==== Noise ====
        if self.use_zeros_start:
            noise = torch.zeros_like(image_latents)
        else:
            noise = torch.randn_like(image_latents)
        latents = noise
        
        # ==== Denoise ====
        for t in tqdm(self.noise_scheduler.timesteps):
            timestep = t.expand(batch_size).to(self.device)
            transformer_input = torch.cat([image_latents, latents], dim=1)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.transformer(
                    hidden_states=transformer_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_batch,
                    pooled_projections=pooled_prompt_embeds_batch,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, t, latents).prev_sample
            if self.use_zeros_start:
                latents = noise_pred
            else:
                latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents, noise
