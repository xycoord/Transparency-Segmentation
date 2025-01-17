import copy
import math
import torch
from tqdm import tqdm
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import get_logger

import logging
import transformers
import diffusers
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import compute_loss_weighting_for_sd3, free_memory

from dataloaders.dataset_configuration import get_trans10k_train_loader, get_trans10k_val_loader

from utils.utils import load_prompt_embeds, compute_max_train_steps
from utils.checkpoint_utils import resume_from_checkpoint
from utils.args_parser import parse_args
from utils.stable_diffusion_3 import sample_timesteps, get_noise_ratio

from log_val import log_validation

logger = get_logger(__name__)

def main():

    args = parse_args() # config.yaml

    # ==== Setup Directories ====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ==== Accelerator ====
    # The accelerator (Hugging Face) takes care of:
    # - Multi-GPU training
    # - Logging
    # - Tracking
    accelerator = Accelerator(
        mixed_precision=args.data_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
    )

    # ==== Mixed Precision ====
    half_dtype = args.torch_dtype
    full_dtype = torch.float32
    device = accelerator.device

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

    # ==== Tracking ==== 
    # Trackers (Weights and Biases) record metrics such as loss and learning rate during training.
    # Weight and Biases lets us visualize these in a web interface.
    accelerator.init_trackers(
        "sd3-finetune-transparency",
        config={
        "dataset": "trans10k",
        "epochs": args.epochs,
        "batch_size": args.train_batch_size,
        "learning_rate": args.lr,
        "num_warmup_steps": args.lr_warmup_steps,
        "num_cycles": args.lr_cycles,
        "max_grad_norm": args.max_grad_norm,
        "num_processes": accelerator.num_processes,
        }
    )


    # ======== LOAD MODELS ========

    # ==== Load Precomputed Prompt Embeddings ====
    # For this task, we condition the transformer with empty prompts ("")
    # Instead of loading the tokenizers and text encoders, we directly load the precomputed prompt embeddings.
    # This save time and memory.
    # See the script `compute_empty_prompt.py` for how these are computed.
    prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(args.prompt_embeds_path)
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    # ==== Load VAE ====
    # The VAE is a pretrained model which we will use to encode images and masks into the latent space.
    vae = AutoencoderKL.from_pretrained(
        args.base_model_path, 
        subfolder="vae", 
        torch_dtype=half_dtype, 
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
        torch_dtype=full_dtype, # Load with full precision for safety
        low_cpu_mem_usage=False,
        in_channels=32, # 16 for masks + 16 for images 
        out_channels=16,
        # We've made a modification to the architecture so need to tell Diffusers that this is intended.
        ignore_mismatched_sizes=True,
        # Sample size and qk norm use these values anyway, but we set them explicitly to be clear
        sample_size=128,
        qk_norm="rms_norm"
    )
    transformer.requires_grad_(True) # We are training the transformer
    # Gradient checkpointing is essential for avoiding OOM errors.
    transformer.enable_gradient_checkpointing() 
    logger.info("Transformer Loaded")


    # ==== Noise Scheduler ====
    # The noise scheduler used with SD3(.5)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    logger.info("Noise Scheduler Loaded")

        
    # ======== DATA LOADERS ======== 
    # Dataset: Trans10k
    # See dataset_configuration.py and dataloader_trans10k.py for how the dataset is loaded.
    # This is a streamlined version of the original Trans10k dataloader
    # Noteably, values for masks are either 0 or 1 and the "Things"/"Stuff" distinction is removed.
    with accelerator.main_process_first():
        train_loader = get_trans10k_train_loader(
                        args.dataset_path, 
                        batch_size=args.train_batch_size, # Batch given Per GPU
                        logger=logger)
        val_loader = get_trans10k_val_loader(
                        args.dataset_path, 
                        difficulty='mix', # TODO: Use args here
                        logger=logger)
    

    # ======== LEARNING RATE AND OPTIMIZER ========

    # ==== Scale Learning Rate ====
    # When the effective batch size is changed the learning rate must be scaled accordingly.
    # The lr set in the args is the learning rate for a batch of 1.
    prescaled_lr = args.lr
    args.lr = prescaled_lr * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    logger.info(f"Learning Rate Scaled: {prescaled_lr} -> {args.lr}")

    # ==== Optimizer ====
    # Following the SD3 paper, epsilon is set to 1e-15 in the args.
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info("Optimizer Initialized with AdamW")

    # ==== Learning Rate Scheduler ====

    args.max_train_steps, overrode_max_train_steps = compute_max_train_steps(
        len(train_loader),
        args.epochs,
        args.gradient_accumulation_steps,
        logger=logger,
        max_train_steps=args.max_train_steps,
    )
    # There is a bug in Diffusers with get_schedule so we directly use the cosine schedule.
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes,
        num_cycles=args.lr_cycles * accelerator.num_processes,
        num_training_steps = args.max_train_steps * accelerator.num_processes,
    )
    logger.info("Learning Rate Scheduler Initialized")


    # ======== Prepare all with Accelerator ========
    # The accelerator wraps the components to handle multi-GPU training.
    transformer, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        transformer, optimizer, lr_scheduler, train_loader, val_loader
    )
    logger.info("Accelerator Prepared")
    
    # Preparing the loaders will change their length, so we need to update the number of steps.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = num_update_steps_per_epoch * args.epochs
    # Automatically set val steps
    args.val_steps = math.ceil(num_update_steps_per_epoch / 2) 

        
    # ======== Resume from Checkpoint ========
    # Checkpoints are stored in the output directory under the "checkpoints" folder.
    # To load the latest checkpoint, set the resume_from_checkpoint argument to "latest".
    # To load a specific checkpoint, set the resume_from_checkpoint argument to the name of the checkpoint.
    # TODO should I abstract this all to resume_from_checkpoint?
    if args.resume_from_checkpoint:
        initial_global_step = resume_from_checkpoint(args.resume_from_checkpoint, checkpoint_dir, accelerator, logger)

        # ==== Force new learning rate ====
        # modify the deepspeed optimizer wrapper
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = args.lr
            param_group['lr'] = args.lr

        # modify the base scheduler's parameters
        lr_scheduler.scheduler.base_lrs = [args.lr]

    else:
        logger.info("Starting a new training run.")
        initial_global_step = 0

    global_step = initial_global_step
    first_epoch = global_step // num_update_steps_per_epoch


    # ==== Print Training Info ====
    logger.info(f"Number of Update Steps per Epoch: {num_update_steps_per_epoch}")
    logger.info(f"Val Steps: {args.val_steps}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Initial Global Step: {initial_global_step}")
    logger.info(f"Max Train Steps: {args.max_train_steps}")

    # This doesn't hold because checkpoints might be saved mid gradient accumulation step.
    # steps_in_current_epoch = global_step % num_update_steps_per_epoch
    # logger.info(f"Steps in Current Epoch: {steps_in_current_epoch}")
    # train_loader = accelerator.skip_first_batches(train_loader, steps_in_current_epoch)


    # ======== TRAINING LOOP ========
    
    for epoch in range(first_epoch, args.epochs):
        transformer.train()
        accum_loss = 0.0
        accum_steps = 0

        logger.info(f"Epoch {epoch}")

        progress_bar = tqdm(
            range(0, num_update_steps_per_epoch),
            initial=0,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        for batch in train_loader:
            with accelerator.accumulate(transformer):
                images, masks, names = batch 
                batch_size = masks.shape[0]

                # ==== Copy single channel mask across RGB channels ====
                masks_stacked = masks.unsqueeze(1).repeat(1,3,1,1).float() # dim 0 is batch?

                # ==== Preprare for Encoder ====
                images_normalized = vae_image_processor.normalize(images).to(device).to(half_dtype)
                masks_normalized= vae_image_processor.normalize(masks_stacked).to(device).to(half_dtype)
                
                # ==== Encode ====
                image_latents = vae.encode(images_normalized).latent_dist.sample()
                mask_latents = vae.encode(masks_normalized).latent_dist.sample()

                # ==== Prepare for Transformer ====
                # Scale so latents behave well with diffusion
                image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                mask_latents = (mask_latents - vae.config.shift_factor) * vae.config.scaling_factor

                # ==== Noise ====
                noise = torch.randn_like(mask_latents)
                timesteps = sample_timesteps(
                    noise_scheduler=noise_scheduler_copy, 
                    batch_size=batch_size, 
                    weighting_scheme=args.weighting_scheme,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                ).to(device)

                noise_ratio = get_noise_ratio(timesteps, noise_scheduler_copy, accelerator, n_dim=mask_latents.ndim, dtype=half_dtype)
                # Add noise according to rectified flow.
                noisy_mask_latents = (1.0 - noise_ratio) * mask_latents + noise_ratio * noise

                transformer_input = torch.cat([image_latents, noisy_mask_latents], dim=1)

                # ==== Prepare Prompt Embeds ====
                prompt_embeds_batch = prompt_embeds.repeat(batch_size, 1, 1)
                pooled_prompt_embeds_batch = pooled_prompt_embeds.repeat(batch_size, 1)

                # ==== Forward Pass ====
                model_pred = transformer(
                        hidden_states=transformer_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_batch,
                        pooled_projections=pooled_prompt_embeds_batch,
                        return_dict=False,
                    )[0]

                target = noise - mask_latents

                # Compute loss.
                # TODO Explain this loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=noise_ratio)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Accumulate for metrics
                accum_loss += loss.detach().item()
                accum_steps += 1

            if accelerator.sync_gradients:
                logger.debug(f"Syncing Gradients")
                progress_bar.update(1)
                global_step += 1

                gn = transformer.get_global_grad_norm()
                grad_norm = gn.item() if gn is not None else 0.0

                logs = {"loss": accum_loss/accum_steps, "lr": lr_scheduler.get_last_lr()[0], "grad_norm": grad_norm}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                accum_loss = 0.0
                accum_steps = 0

                # ==== Save checkpoint ====
                # if global_step % args.save_checkpoint_steps == 0 and global_step > 0:
                #     accelerator.wait_for_everyone()
                #     accelerator.save_state(checkpoint_dir / f"checkpoint-{global_step}")
                
                # ==== Validation ====
                if global_step % args.val_steps == 0 and global_step > 0:
                    
                    free_memory()

                    transformer.eval()
                    log_validation( 
                        args=args,
                        accelerator=accelerator,
                        vae=vae,
                        transformer=accelerator.unwrap_model(transformer),
                        noise_scheduler=accelerator.unwrap_model(noise_scheduler),
                        data_loader=val_loader,
                        global_step=global_step,
                        denoise_steps=20,
                        num_vals=4,
                        ensemble_size=5,
                        logger=logger,
                        )
                    transformer.train()
                    logger.info("Validation Done")

                    free_memory()

        accelerator.wait_for_everyone()
        # NOTE gradient accumulation steps may cross epoch boundaries so saving at the end of the epoch is not ideal.
        accelerator.save_state(checkpoint_dir / f"checkpoint-{global_step}")
                
    accelerator.end_training()
                             

if __name__=="__main__":
    main()
