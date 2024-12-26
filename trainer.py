import torch
from args_parser import parse_args

from accelerate import Accelerator

from accelerate.logging import get_logger
import logging

from dataset_configuration import prepare_dataset
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler

def main():

    args = parse_args()

    # Set up accelerator
    accelerator = Accelerator(
        mixed_precision="bf16", # could be argument
    )

    dtype = torch.bfloat16 # could be argument (align with mixed_precision)
    device = accelerator.device
    logger.info(f"Dtype: {dtype}")
    logger.info(f"Device: {device}")

    # Set up logging (accelerate, wandb)

    # Set up output directory

    '''--Non-NN Modules Definiton--'''
    # Tokenizer (can I precompute the)
    # Load Text Encoder (requires_grad=False)

    # Load VAE 
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)
    vae.eval() # Does accelerator handle this?
    vae_image_processor = VaeImageProcessor(do_normalize=True)
    logger.info("VAE Loaded")

    # Load Transformer (train())
    transformer = SD3Transformer2DModel.from_pretrained(
        args.base_model_path, 
        subfolder="transformer",
        use_safetensors=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    logger.info("Transformer Loaded")


    # EMA Transformer? (EMA Unet)

    # ==== Noise Scheduler ====

    # Xformers

    # Gradient Checkpointing?

    # Learning rate
        # Consider batch size, number of GPUs, and gradient accumulation steps

    # ==== Initialize the optimizer ====
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ==== Learning Rate Scheduler ====
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps = args.max_train_steps * accelerator.num_processes,
    )    

    # ==== Data Loaders ==== 
    with accelerator.main_process_first():
        (train_loader, val_loader, test_loader), dataset_config_dict = prepare_dataset(
            data_name=args.dataset_name,
            dataset_path=args.dataset_path,
            batch_size=args.train_batch_size,
            test_batch=1,
            datathread=args.dataloader_num_workers,
            logger=logger)

    # ==== Prepare all with Accelerator ====
    transformer, optimizer, train_loader, test_loader, val_loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_loader, test_loader, val_loader, lr_scheduler
    )

    # Precision (Mixed Precision?)

    # Resume from checkpoint

    # ==== Training Loop ====
    # Validate regularly

if __name__=="__main__":
    main()
