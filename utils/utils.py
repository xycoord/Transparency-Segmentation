import math
import torch
from pathlib import Path

def load_prompt_embeds(path):
    prompt_embeds_path = Path(path)/"empty_prompt_embeds.pt"
    if not prompt_embeds_path.exists():
        raise FileNotFoundError(f"Prompt Embeddings not found at {prompt_embeds_path}")
    pooled_prompt_embeds_path = Path(path)/"empty_pooled_prompt_embeds.pt"
    if not pooled_prompt_embeds_path.exists():
        raise FileNotFoundError(f"Pooled Prompt Embeddings not found at {pooled_prompt_embeds_path}")
    return torch.load(prompt_embeds_path, weights_only=True), torch.load(pooled_prompt_embeds_path, weights_only=True)
    # Set weights_only=True to supress a security warning.

def compute_max_train_steps(batches_per_epoch, epochs, gradient_accumulation_steps, logger=None, max_train_steps=None):
    num_update_steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    overrode = False
    if max_train_steps is None:
        max_train_steps = epochs * num_update_steps_per_epoch
        overrode = True
        if logger is not None:
            logger.info(f"Max Train Steps Computed: {max_train_steps}")
    elif logger is not None:
        logger.info(f"Max Train Steps Provided: {max_train_steps}")
    return max_train_steps, overrode

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")