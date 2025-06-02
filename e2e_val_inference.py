import torch
from pathlib import Path
import os

from accelerate import Accelerator
from accelerate.logging import get_logger
import logging

import transformers
import diffusers
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
import torch.serialization

from utils.args_parser import parse_args
from dataloaders.dataset_configuration import get_trans10k_test_loader, get_trans10k_val_loader
from utils.checkpoint_utils import get_checkpoint_path, get_global_step_from_checkpoint
from e2e_log_val import log_validation


logger = get_logger(__name__)

def main():

    CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])

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
        mixed_precision=args.data_type, # necessary for inference?
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
    # accelerator.init_trackers(
    #     "sd3-finetune-transparency",
    #     config={
    #     "dataset": "trans10k",
    #     "val_batch_size": args.test_batch_size,
    #     "num_processes": accelerator.num_processes,
    #     }
    # )

    # ======== LOAD MODELS ========

    # ==== Load VAE ====
    # The VAE is a pretrained model which we will use to encode images and masks into the latent space.
    vae = AutoencoderKL.from_pretrained(
        args.base_model_path, 
        subfolder="vae", 
        torch_dtype=half_dtype, 
        use_safetensors=True
    )
    vae = vae.to(device).requires_grad_(False)
    vae.eval() 
    vae_image_processor = VaeImageProcessor(do_normalize=True)
    logger.info("VAE Loaded")

    # ==== Load Transformer ====
    transformer = SD3Transformer2DModel.from_pretrained(
        args.base_model_path, 
        subfolder="transformer",
        # use_safetensors=True,
        torch_dtype=full_dtype, # Load with full precision for safety
        low_cpu_mem_usage=False,
        in_channels=16,
        out_channels=16,
        # Sample size and qk norm use these values anyway, but we set them explicitly to be clear
        sample_size=128,
        qk_norm="rms_norm"
    )
    transformer.eval()
    transformer.requires_grad_(False)
    logger.info("Transformer Loaded")

    # ======== DATA LOADERS ======== 
    with accelerator.main_process_first():
        val_loader = get_trans10k_test_loader(args.dataset_path, difficulty=args.test_difficulty, batch_size=4, logger=logger)
        

    # ======== Prepare all with Accelerator ========
    # The accelerator wraps the components to handle multi-GPU training.
    val_loader = accelerator.prepare(val_loader)
    logger.info("Accelerator Prepared")
 
    # ======== Resume from Checkpoint ========
    # Checkpoints are stored in the output directory under the "checkpoints" folder.
    # To load the latest checkpoint, set the resume_from_checkpoint argument to "latest".
    # To load a specific checkpoint, set the resume_from_checkpoint argument to the name of the checkpoint.

    if args.load_checkpoint:
        checkpoint_name = args.load_checkpoint
        checkpoint_path = get_checkpoint_path(checkpoint_name, checkpoint_dir)

        if checkpoint_path is None or not checkpoint_path.exists():
            logger.info(f"checkpoint '{checkpoint_name}' does not exist. terminating.")
            return
        else: # load checkpoint
            accelerator.wait_for_everyone()
            torch.serialization.add_safe_globals([
                ZeroStageEnum,
                LossScaler
            ])
            fp32_model = load_state_dict_from_zero_checkpoint(transformer, checkpoint_path)
            transformer = accelerator.prepare_model(fp32_model)
            transformer.to(half_dtype)
            global_step = get_global_step_from_checkpoint(checkpoint_path)
    else:
        logger.info("no checkpoint specified. terminating.")
        return

    # transformer = transformer.to(half_dtype).to(device)
    # global_step = 0

    subpath = f"test_{args.test_difficulty}_{global_step}"

    log_validation( 
        args=args,
        accelerator=accelerator,
        vae=vae,
        transformer=transformer,
        data_loader=val_loader,
        subpath=subpath,
        num_vals=args.num_vals_test,
        # num_vals=32,
        logger=logger,
    )

    accelerator.end_training()


if __name__=="__main__":
    main()