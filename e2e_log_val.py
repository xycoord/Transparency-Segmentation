from e2e_prediction_pipeline import PredictionPipeline
from pathlib import Path
from tqdm import tqdm

from utils.image_saver import AsyncImageSaver
from utils.image_conversion import tensor_to_pil
from utils.metrics import convert_batch_metrics, intersection_over_union, recall, mse, accumulate_metrics, get_average_metrics


def log_validation(args, accelerator, vae, transformer, data_loader, subpath, global_step=0, num_vals=None, logger=None):

    device = accelerator.device

    pipeline = PredictionPipeline(
        transformer=transformer,
        vae=vae,
    )
    pipeline.to(device)

    image_output_dir = Path(args.output_dir) / "mask_predictions" / subpath
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    effective_batch_size = data_loader.total_batch_size

    total_items = len(data_loader) * effective_batch_size
    total_items = total_items if not num_vals else min(total_items, num_vals)  

    accumulated_metrics = None

    with tqdm(total=total_items, desc="Validation", disable=not accelerator.is_local_main_process) as progress_bar:
        saver = AsyncImageSaver(num_workers=4)  # Adjust number of workers as needed
        try:
            for batch_index, batch in enumerate(data_loader):
                if num_vals and batch_index * effective_batch_size >= num_vals:
                    break 
                images, masks, names = batch

                ids = [name.split('.')[0] for name in names]

                # Duplicate masks to across RGB channels
                masks_stacked = masks.unsqueeze(1).repeat(1,3,1,1).float().to(device)

                prediction = pipeline(images)

                # ==== Gather ====
                mask_pt, mask_q_pt, mask_gt_pt, image_pt = accelerator.gather_for_metrics((
                    prediction.mask_pt,
                    prediction.maskq_pt,
                    masks_stacked,
                    images
                ))
                mask_pil = accelerator.gather_for_metrics(prediction.mask_pil)
                mask_q_pil = accelerator.gather_for_metrics(prediction.maskq_pil)
                ids = accelerator.gather_for_metrics(ids)
                gathered_batch_size = mask_pt.shape[0]

                if accelerator.is_local_main_process: 
                    # Calculate metrics
                    metrics = {
                        "iou": intersection_over_union(mask_q_pt, mask_gt_pt),
                        "recall": recall(mask_q_pt, mask_gt_pt),
                        "mse": mse(mask_pt, mask_gt_pt),
                    }
                    metrics_list = convert_batch_metrics(gathered_batch_size, metrics)

                    # Queue mask predictions with their corresponding metrics
                    for (id, mask_pred, item_metrics) in zip(ids, mask_pil, metrics_list):
                        path = image_output_dir / f"{id}_pred.png"
                        saver.save([mask_pred], [path], [item_metrics])  # Pass metrics as metadata
                
                    # Queue ground truth masks and images 
                    mask_gt_pil = tensor_to_pil(mask_gt_pt)
                    images_pil = tensor_to_pil(image_pt)
                    for postfix, image_list in [
                        ("maskgt", mask_gt_pil),
                        ("img", images_pil),
                        ("maskq", mask_q_pil),
                    ]:
                        paths = [
                            image_output_dir / f"{id}_{postfix}.png"
                            for id in ids 
                        ]
                        saver.save(image_list, paths)

                    accumulated_metrics = accumulate_metrics(metrics, accumulated_metrics)

                progress_bar.update(effective_batch_size)
        finally:
            saver.close()  # Ensure workers are properly shut down

    # Calculate average metrics
    if accelerator.is_local_main_process:

        average_metrics = get_average_metrics(accumulated_metrics)
        logger.info(f"IOU: {average_metrics['iou']:.4f}")
        logger.info(f"MSE: {average_metrics['mse']:.4f}")
        logger.info(f"Recall: {average_metrics['recall']:.4f}")
        accelerator.log(average_metrics, step=global_step)

    del pipeline