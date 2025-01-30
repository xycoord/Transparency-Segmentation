import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float = 0.5, 
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to a minimum learning rate, after a warmup period during which it increases 
    linearly between 0 and the initial lr set in the optimizer.
    
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to min_lr following a half-cosine).
        min_lr_ratio (`float`, *optional*, defaults to 0.0):
            Minimum learning rate ratio relative to the initial learning rate. Should be between 0 and 1.
            For example, 0.1 means the minimum learning rate will be 10% of the initial learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be between 0 and 1, got {min_lr_ratio}")
        
    def lr_lambda(current_step):
        # Handle warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Calculate progress after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # Calculate cosine decay with minimum lr
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        
        # Scale the decay to respect minimum learning rate
        return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr_range_test_scheduler(optimizer, initial_lr, final_lr, total_iters):
    """
    Creates a LambdaLR scheduler that performs the LR range test.
    
    Args:
        optimizer: The optimizer to use
        initial_lr: Starting learning rate
        final_lr: Ending learning rate
        num_epochs: Number of epochs to run the test
        
    Returns:
        LambdaLR scheduler
    """
    # We need to calculate the multiplier that will get us from initial_lr to final_lr
    # Since LambdaLR multiplies the initial_lr by our lambda function's output,
    # we need to structure the multiplier accordingly
        
    total_mult = final_lr / initial_lr
    def lr_lambda(iteration):
    # Linear increase from 1 to total_mult over all iterations
        return 1 + (total_mult - 1) * (iteration / total_iters)
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

from matplotlib import pyplot as plt

def plot_lr_schedule(scheduler, num_steps, accelerator=None):
    """
    Plot learning rate schedule for a given PyTorch scheduler
    
    Args:
        scheduler: PyTorch learning rate scheduler
        num_steps: Number of steps to plot
    """
    lrs = []
    for i in range(num_steps):
        lrs.append(scheduler.get_last_lr()[-1])
        scheduler.step()

    if accelerator is None or accelerator.is_main_process:
        plt.figure(figsize=(9, 5))
        plt.plot(range(num_steps), lrs)
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.show()