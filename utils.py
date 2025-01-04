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
    if max_train_steps is None:
        max_train_steps = epochs * num_update_steps_per_epoch
        if logger is not None:
            logger.info(f"Max Train Steps Computed: {max_train_steps}")
    elif logger is not None:
        logger.info(f"Max Train Steps Provided: {max_train_steps}")
    return max_train_steps


from diffusers.training_utils import compute_density_for_timestep_sampling

def sample_timesteps(noise_scheduler, batch_size, weighting_scheme, logit_mean, logit_std, mode_scale=None):
    # Sample a random timestep for each image
    # for weighting schemes where we sample timesteps non-uniformly
    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices]
    return timesteps

def get_noise_ratio(timesteps, noise_scheduler, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


