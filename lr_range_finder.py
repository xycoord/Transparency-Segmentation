import torch.optim as optim

from utils.testing_plot_lr_schedule import plot_lr_schedule

def create_lr_range_test_scheduler(optimizer, initial_lr, final_lr, total_iters):
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
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

#test
# import torch
# dummy_param = torch.nn.Parameter(torch.randn(1))
# dummy_optimizer = optim.SGD([dummy_param], lr=1e-7) 

# # Create the scheduler for range test from 1e-7 to 1e-2 over 4 epochs
# scheduler = create_lr_range_test_scheduler(
#     optimizer=dummy_optimizer,
#     initial_lr=1e-7,
#     final_lr=1e-2,
#     total_iters=200 
# )

# plot_lr_schedule(scheduler, num_steps=200)