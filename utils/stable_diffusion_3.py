import torch
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

    # Timestep Shifting as per paper
    alpha = 3.0  # As recommended in paper for 1024x1024

    # After getting initial timesteps
    t_n = timesteps.float() / noise_scheduler.config.num_train_timesteps
    t_m = (alpha * t_n) / (1 + (alpha - 1) * t_n)
    timesteps = (t_m * noise_scheduler.config.num_train_timesteps).to(torch.float32)

    # Find closest timesteps in noise_scheduler.timesteps
    timesteps = torch.tensor([noise_scheduler.timesteps[torch.abs(noise_scheduler.timesteps - t).argmin()] for t in timesteps])

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
