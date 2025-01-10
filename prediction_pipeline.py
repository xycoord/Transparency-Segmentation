from diffusers import DiffusionPipeline, AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
import numpy as np
import torch
from PIL import Image
from typing import Union

from tqdm import tqdm

from utils import load_prompt_embeds
from diffusers.image_processor import VaeImageProcessor


class MaskPipelineOutput(BaseOutput):
    mask_np: torch.Tensor
    mask_colored: Image.Image
    mask_vae: Image.Image
    noise_output: Image.Image
    uncertainty: Union[None, np.ndarray]


class PredictionPipeline(DiffusionPipeline):

    def __init__(self,
                 transformer:SD3Transformer2DModel,
                 vae:AutoencoderKL,
                 noise_scheduler:FlowMatchEulerDiscreteScheduler,
                 ):
        super().__init__()
            
        self.register_modules(
            transformer=transformer,
            vae=vae,
            noise_scheduler=noise_scheduler,
        )

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
                 processing_res: int = 1024,
                 match_input_res:bool =True,
                 batch_size:int =1,
                 color_map: str="gray",
                 show_progress_bar:bool = True,
                 ) -> MaskPipelineOutput:

        # ==== Prepare Prompt Embeds ====
        prompt_embeds_batch = self.prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds_batch = self.pooled_prompt_embeds.repeat(batch_size, 1)

        # ==== Set Timesteps ====
        self.noise_scheduler.set_timesteps(denoise_steps, device=self.device)

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

        output_latents, noise = self.generate_mask(batch_size, image_latents, prompt_embeds_batch, pooled_prompt_embeds_batch)

        # decode the output
        output_latents = (output_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        mask_latents = (mask_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        noise = (noise / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.no_grad():
            output = self.vae.decode(output_latents).sample
            mask = self.vae.decode(mask_latents).sample
            noise = self.vae.decode(noise).sample


        mask_pred = self.vae_image_processor.postprocess(output, output_type='pt')
        output_PIL = self.vae_image_processor.postprocess(output, output_type='pil')
        mask_pt = self.vae_image_processor.postprocess(mask, output_type='pt')
        mask_PIL = self.vae_image_processor.postprocess(mask, output_type='pil')
        noise_PIL = self.vae_image_processor.postprocess(noise, output_type='pil')


        return MaskPipelineOutput(mask_pt=mask_pred, mask_colored=output_PIL, mask_vae=mask_PIL, noise_output=noise_PIL, uncertainty=None)
    


    def generate_mask(self, batch_size, image_latents, prompt_embeds_batch, pooled_prompt_embeds_batch):
        # ==== Noise ====
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
            latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents, noise
