import copy
import math
from checkpoint_utils import get_checkpoint_path, get_global_step_from_checkpoint
import torch
from tqdm import tqdm
from args_parser import parse_args
import os
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from accelerate.logging import get_logger
import logging
import transformers
import diffusers
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import is_compiled_module

from dataset_configuration import prepare_dataset
from utils import compute_max_train_steps, load_prompt_embeds, get_noise_ratio, sample_timesteps

logger = get_logger(__name__)

def main():

    args = parse_args()

    # ==== Setup Directories ====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True) # Logging directory doesn't seem to be used
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ==== Accelerator ====
    accelerator = Accelerator(
        mixed_precision=args.data_type,
    )


    # ==== Logging ====
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    

    # Set up WandB logging 



    # Mixed Precision
    inf_dtype = args.torch_dtype # VAE etc.
    train_dtype = torch.float32 # Transformer etc.
    device = accelerator.device
    logger.info(f"Inference Data Type: {inf_dtype}")
    logger.info(f"Training Data Type: {train_dtype}")
    logger.info(f"Device: {device}")


    # ======== LOAD MODELS ========

    # ==== Load Precomputed Prompt Embeddings ====
    prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(args.prompt_embeds_path)
    prompt_embeds = prompt_embeds.to(train_dtype).to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(train_dtype).to(device)

    # ==== Load VAE ====
    vae = AutoencoderKL.from_pretrained(
        args.base_model_path, 
        subfolder="vae", 
        torch_dtype=inf_dtype, 
        use_safetensors=True
    )
    vae = vae.to(device).requires_grad_(False)
    vae.eval() # We're not training the VAE
    vae_image_processor = VaeImageProcessor(do_normalize=True)
    logger.info("VAE Loaded")

    # ==== Load Transformer ====
    transformer = SD3Transformer2DModel.from_pretrained(
        args.base_model_path, 
        subfolder="transformer",
        # use_safetensors=True,
        torch_dtype=train_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        in_channels=32, # 16 for masks + 16 for images 
        sample_size=128
    )
    transformer.requires_grad_(True)
    transformer.enable_gradient_checkpointing()
    logger.info("Transformer Loaded")

    # EMA Transformer? (EMA Unet)

    # ==== Noise Scheduler ====
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    logger.info("Noise Scheduler Loaded")

        
    # ======== DATA LOADERS ======== 

    with accelerator.main_process_first():
        (train_loader, val_loader, test_loader), dataset_config_dict = prepare_dataset(
            data_name=args.dataset_name,
            dataset_path=args.dataset_path,
            batch_size=args.train_batch_size,
            test_batch=1, # args.test_batch_size, what is this?
            datathread=args.dataloader_num_workers,
            logger=logger)
    
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)


    # ======== LEARNING RATE AND OPTIMIZER ========

    # ==== Scale Learning Rate ====
    prescaled_lr = args.lr
    args.lr = prescaled_lr * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    logger.info(f"Learning Rate Scaled: {prescaled_lr} -> {args.lr}")

    # ==== Optimizer ====
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info("Optimizer Initialized with AdamW")

    # ==== Learning Rate Scheduler ====
    args.max_train_steps = compute_max_train_steps(
        len(train_loader),
        args.epochs,
        args.gradient_accumulation_steps,
        logger=logger,
        max_train_steps=args.max_train_steps,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps = args.max_train_steps * accelerator.num_processes,
    )
    logger.info("Learning Rate Scheduler Initialized")


    # ======== Prepare all with Accelerator ========
    transformer, optimizer, train_loader, test_loader, val_loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_loader, test_loader, val_loader, lr_scheduler
    )
    logger.info("Accelerator Prepared")


    # ======== Resume from Checkpoint ========
    if args.resume_from_checkpoint:
        checkpoint_name = args.resume_from_checkpoint
        checkpoint_path = get_checkpoint_path(checkpoint_name, checkpoint_dir)

        if checkpoint_path is None or not checkpoint_path.exists():
            logger.info(f"Checkpoint '{checkpoint_name}' does not exist. Starting a new training run.")
            initial_global_step = 0
        else: # Load Checkpoint
            accelerator.wait_for_everyone()
            accelerator.load_state(checkpoint_path)
            global_step = get_global_step_from_checkpoint(checkpoint_path)
            initial_global_step = global_step
    else:
        logger.info("Starting a new training run.")
        initial_global_step = 0

    global_step = initial_global_step
    first_epoch = global_step // num_update_steps_per_epoch
    steps_in_current_epoch = global_step % num_update_steps_per_epoch

    train_loader = accelerator.skip_first_batches(train_loader, steps_in_current_epoch)
                

    # ======== TRAINING LOOP ========

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.epochs):
        transformer.train()
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                images, masks, names = batch 
                batch_size = masks.shape[0]
                
                # ==== Reshape data for stable diffusion standards ====
                masks_stacked = masks.unsqueeze(1).repeat(1,3,1,1).float() # dim 0 is batch?

                # ==== Preprare for Encoder ====
                images_normalized = vae_image_processor.normalize(images).to(device).to(inf_dtype)
                masks_normalized= vae_image_processor.normalize(masks_stacked).to(device).to(inf_dtype)
                
                # ==== Encode ====
                image_latents = vae.encode(images_normalized).latent_dist.sample()
                mask_latents = vae.encode(masks_normalized).latent_dist.sample()

                # ==== Prepare for Transformer ====
                # Scale so latents behave well with diffusion
                image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                mask_latents = (mask_latents - vae.config.shift_factor) * vae.config.scaling_factor
                # Transformer is trained with full precision
                image_latents = image_latents.to(train_dtype)
                mask_latents = mask_latents.to(train_dtype)

                # ==== Noise ====
                noise = torch.randn_like(mask_latents)
                timesteps = sample_timesteps(
                    noise_scheduler=noise_scheduler_copy, 
                    batch_size=batch_size, 
                    weighting_scheme=args.weighting_scheme,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale
                ).to(device)

                noise_ratio = get_noise_ratio(timesteps, noise_scheduler_copy, accelerator, n_dim=mask_latents.ndim, dtype=train_dtype)
                # Add noise according to flow matching.
                noisy_mask_latents = (1.0 - noise_ratio) * mask_latents + noise_ratio * noise


                # ==== Forward Pass ====
                transformer_input = torch.cat([image_latents, noisy_mask_latents], dim=1)
                transformer_input = transformer_input.to(torch.bfloat16)
                timesteps = timesteps.to(torch.bfloat16)
                prompt_embeds = prompt_embeds.to(torch.bfloat16)
                pooled_prompt_embeds = pooled_prompt_embeds.to(torch.bfloat16)

                model_pred = transformer(
                        hidden_states=transformer_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                logger.info(f"Forward Pass Done")

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=noise_ratio)
                target = noise - mask_latents
                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                logger.info(f"Loss Computed")

                accelerator.backward(loss)
                logger.info(f"Backward Pass Done")

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                logger.info(f"Gradient Clipped")

                optimizer.step()
                logger.info(f"Optimizer Step Done")
                lr_scheduler.step()
                logger.info(f"LR Scheduler Step Done")
                optimizer.zero_grad()
                logger.info(f"Optimizer Zero Grad Done")

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # ==== Save checkpoint ====
                if global_step % args.save_checkpoint_steps == 0 and global_step > 0:
                    accelerator.wait_for_everyone()
                    accelerator.save_state(checkpoint_dir / f"checkpoint-{global_step}")

                # ==== Validation ====


    accelerator.end_training()
                             

if __name__=="__main__":
    main()
