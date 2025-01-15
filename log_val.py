from diffusers.image_processor import VaeImageProcessor
from prediction_pipeline import PredictionPipeline
from pathlib import Path


def log_validation(args, accelerator, vae, transformer, noise_scheduler, data_loader, global_step, denoise_steps=50, num_vals=10, ensemble_size=10, logger=None):

    device = accelerator.device

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

        # Duplicate masks to across RGB channels
        masks_stacked = masks.unsqueeze(1).repeat(1,3,1,1).float()
         
        prediction = pipeline(
                        images, 
                        masks_stacked, 
                        denoise_steps=denoise_steps, 
                        ensemble_size=ensemble_size, 
                        batch_size=1, #TODO: use args here NOTE I've found that although greater batch sizes are are fine for memory, 1 seems to be fastest
                     )

        masks_gt = vae_image_processor.pt_to_numpy(masks_stacked)
        masks_gt = vae_image_processor.numpy_to_pil(masks_gt)
        images_gt = vae_image_processor.pt_to_numpy(images)
        images_gt = vae_image_processor.numpy_to_pil(images_gt)

        for i, mask_gt in enumerate(masks_gt):
            mask_gt.save(image_output_dir / f"val_{val_index}_{i}_gt_{accelerator.local_process_index}.png")
        
        for i, image_gt in enumerate(images_gt):
            image_gt.save(image_output_dir / f"val_{val_index}_{i}_img_{accelerator.local_process_index}.png")

        for i, mask_pred in enumerate(prediction.mask_pil):
            mask_pred.save(image_output_dir / f"val_{val_index}_{i}_{accelerator.local_process_index}.png")
        
        for i, mask_v in enumerate(prediction.mask_vae):
            mask_v.save(image_output_dir / f"val_{val_index}_{i}_vae_{accelerator.local_process_index}.png")

    del pipeline
    del vae_image_processor