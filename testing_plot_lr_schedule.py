import torch

import matplotlib.pyplot as plt

def plot_lr_schedule(scheduler, num_steps, accelerator):
    """
    Plot learning rate schedule for a given PyTorch scheduler
    
    Args:
        scheduler: PyTorch learning rate scheduler
        num_steps: Number of steps to plot
    """
    lrs = []
    for i in range(num_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    if accelerator.is_main_process:
        plt.figure(figsize=(10, 5))
        plt.plot(range(num_steps), lrs)
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.show()
