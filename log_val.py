import torch

from diffusers import FlowMatchEulerDiscreteScheduler

from diffusers.image_processor import VaeImageProcessor
from prediction_pipeline import PredictionPipeline
from pathlib import Path


def log_validation(args, accelerator, vae, transformer, noise_scheduler, data_loader, global_step, denoise_steps=50, num_vals=10, ensemble_size=10, logger=None):

    device = accelerator.device
    width = 1024
    height = 1024

    vae_image_processor = VaeImageProcessor(do_normalize=True)

    pipeline = PredictionPipeline(
        transformer=transformer,
        vae=vae,
        noise_scheduler=noise_scheduler,
    )
    pipeline.to(device)

    image_output_dir = Path(args.output_dir) / "mask_predictions" / f"step_{global_step}"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    for val_index, batch in enumerate(data_loader):
        if val_index >= num_vals:
            break

        images, masks, names = batch 
        batch_size = images.shape[0]

        masks_stacked = masks.unsqueeze(1).repeat(1,3,1,1).float() # dim 0 is batch?
         
        prediction = pipeline(images, masks_stacked, denoise_steps=denoise_steps, ensemble_size=ensemble_size, processing_res=width, match_input_res=True, batch_size=batch_size, color_map="gray", show_progress_bar=False)

        masks_gt = vae_image_processor.pt_to_numpy(masks_stacked)
        masks_gt = vae_image_processor.numpy_to_pil(masks_gt)

        for i, mask_gt in enumerate(masks_gt):
            mask_gt.save(image_output_dir / f"val_{val_index}_{i}_gt_{accelerator.local_process_index}.png")

        for i, mask_pred in enumerate(prediction.mask_colored):
            mask_pred.save(image_output_dir / f"val_{val_index}_{i}_{accelerator.local_process_index}.png")
        
        for i, n in enumerate(prediction.noise_output):
            n.save(image_output_dir / f"val_{val_index}_{i}_noise_{accelerator.local_process_index}.png")
        
        for i, mask_v in enumerate(prediction.mask_vae):
            mask_v.save(image_output_dir / f"val_{val_index}_{i}_vae_{accelerator.local_process_index}.png")

    del pipeline
    del vae_image_processor