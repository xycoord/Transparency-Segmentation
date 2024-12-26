import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune SD3.5 for Image Segmentation of Transparent Objects')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--base-model-path', type=str, default='stabilityai/stable-diffusion-3.5-large',
                        help='Hugging Face path to base model (default: stabilityai/stable-diffusion-3.5-large)')
    
    # Dataset
    dataset_name = 'trans10k'
    parser.add_argument('--dataset-name', type=str, default=dataset_name,
                        help='name of dataset (default: trans10k)')
    dataset_path = '/home/xycoord/models/Trans10k/'
    parser.add_argument('--dataset-path', type=str, default=dataset_path,
                        help='path to dataset (default: /home/xycoord/models/Trans10k/)')
    train_batch_size = 1
    parser.add_argument('--train-batch-size', type=int, default=train_batch_size,
                        help='input batch size for training (default: 1)')
    dataloader_num_workers = 4
    parser.add_argument('--dataloader-num-workers', type=int, default=dataloader_num_workers,
                        help='number of workers for dataloader (default: 4)')

    # Optimizer
    parser.add_argument('--adam-beta1', type=float, default=0.9,
                        help='beta1 for Adam optimizer (default: 0.9)')
    parser.add_argument('--adam-beta2', type=float, default=0.999, 
                        help='beta2 for Adam optimizer (default: 0.999)')
    parser.add_argument('--adam-weight-decay', type=float, default=0.0,
                        help='weight decay for Adam optimizer (default: 0.0)')
    parser.add_argument('--adam-epsilon', type=float, default=1e-15, # 1e-8 default for adam
                        help='epsilon for Adam optimizer (default: 1e-15)')

    # Learning Rate Scheduler
    parser.add_argument('--lr-scheduler', type=str, default='linear',
                        help='learning rate scheduler (default: linear)')
    parser.add_argument('--lr-warmup-steps', type=int, default=1000,
                        help='number of warmup steps for learning rate scheduler (default: 1000)')
    parser.add_argument('--max-train-steps', type=int, default=1000,
                        help='maximum number of training steps (default: 1000)')
    
    


    
    args = parser.parse_args()
    return args