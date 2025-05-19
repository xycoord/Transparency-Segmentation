from PIL import Image

import torch

from diffusers import DiffusionPipeline, AutoencoderKL, SD3Transformer2DModel
from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor

from utils.utils import load_prompt_embeds
from utils.image_conversion import tensor_to_pil


class MaskPipelineOutput(BaseOutput):
    mask_pt: torch.Tensor
    mask_pil: Image.Image
    maskq_pt: torch.Tensor
    maskq_pil: Image.Image

    def __init__(self, mask_pred, mask_pred_q):
        super().__init__()
        self.mask_pt = mask_pred
        self.mask_pil = tensor_to_pil(mask_pred)
        self.maskq_pt = mask_pred_q
        self.maskq_pil = tensor_to_pil(mask_pred_q)
    


class PredictionPipeline(DiffusionPipeline):

    def __init__(self,
                 transformer:SD3Transformer2DModel,
                 vae:AutoencoderKL,
                 ):
        super().__init__()
            
        self.register_modules(
            transformer=transformer,
            vae=vae,
        )

        self.vae_image_processor = VaeImageProcessor(do_normalize=True)

        # ==== Load Precomputed Prompt Embeddings ====
        prompt_embeds_path = './precomputed_prompt_embeddings/'
        prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(prompt_embeds_path)
        self.prompt_embeds = prompt_embeds.to(self.device)
        self.pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)


    @torch.no_grad()
    def __call__(self, input_images: torch.Tensor) -> MaskPipelineOutput:
        
        """
        Process images and masks through the prediction pipeline to generate mask outputs.

        Args:
            input_images (torch.Tensor, optional): Input images to predict from. 

        Returns:
            MaskPipelineOutput: A named tuple containing:
                - mask_pt (torch.Tensor): The predicted masks in PyTorch tensor format
                - mask_pil (PIL.Image): The mask visualization
                - maskq_pt (torch.Tensor): The quantized predicted masks in PyTorch tensor format
                - maskq_pil (PIL.Image): The quantized mask visualization
        """

        batch_size = input_images.shape[0]

        # ==== Prepare Prompt Embeds ====
        prompt_embeds_batch = self.prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds_batch = self.pooled_prompt_embeds.repeat(batch_size, 1)

        # ==== Preprare for Encoder ====
        images_normalized = self.vae_image_processor.normalize(input_images).to(self.device).to(torch.bfloat16) # Shouldn't be hard coded type
        
        # ==== Encode ====
        image_latents = self.vae.encode(images_normalized).latent_dist.sample()
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # Constant timestep
        timestep = torch.ones(batch_size).to(self.device) * 1000.0

        # ==== Make Prediction ====
        with torch.no_grad():
            prediction = self.transformer(
                hidden_states=image_latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds_batch,
                pooled_projections=pooled_prompt_embeds_batch,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        # ==== Decode ====  
        prediction = (prediction / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        prediction = self.vae.decode(prediction).sample

        # ==== Postprocess ====
        mask_pred = self.vae_image_processor.postprocess(prediction, output_type='pt')

        # ==== Quantize ====
        mask_pred_mean = mask_pred.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        mask_pred_quantized = quantize_prediction(mask_pred_mean)

        return MaskPipelineOutput(mask_pred=mask_pred, mask_pred_q=mask_pred_quantized)


def quantize_prediction(tensor, threshold=0.5):
    return (tensor > threshold).float()
