import torch
import numpy as np
from typing import Dict, List


# ==== Metrics ====

def contrast_score(image):
    image_shifted = image - 0.5
    return torch.mean(0.5-torch.abs(image_shifted))

def intersection_over_union(tensor1, tensor2):
    # Compute intersection and union along all dimensions except batch
    intersection = torch.sum((tensor1 * tensor2) > 0, dim=(1, 2, 3))
    union = torch.sum((tensor1 + tensor2) > 0, dim=(1, 2, 3))
    iou = intersection.float() / union.float()
    return iou

def recall(predictions, ground_truth):
    # True positives: where both prediction and ground truth are positive
    true_positive = torch.sum((predictions * ground_truth), dim=(1, 2, 3))
    
    # All actual positives in ground truth
    actual_positives = torch.sum(ground_truth, dim=(1, 2, 3))
    
    # Add small epsilon for numerical stability
    epsilon = 1e-9
    recall = true_positive / (actual_positives + epsilon)
    
    return recall

def mse(predictions, ground_truth):
    return torch.nn.functional.mse_loss(predictions, ground_truth, reduction='none').mean(dim=(1,2,3))


# ==== Utility functions ==== 

def accumulate_metrics(current_metrics, accumulated_metrics=None):
    """
    Accumulate metrics by concatenating tensors.
    
    Args:
        current_metrics (dict): Current batch metrics with PyTorch tensor values
        accumulated_metrics (dict): Previously accumulated metrics, None for first batch
    
    Returns:
        dict: Updated accumulated metrics with concatenated tensors
    """
    if accumulated_metrics is None:
        accumulated_metrics = {
            key: [value.detach().clone()]
            for key, value in current_metrics.items()
        }
    else:
        for key, value in current_metrics.items():
            accumulated_metrics[key].append(value.detach())
            
    return accumulated_metrics

def get_average_metrics(accumulated_metrics):
    """
    Calculate statistics from accumulated tensors.
    """
    return {
        key: torch.cat(values).mean()
        for key, values in accumulated_metrics.items()
    }

def convert_batch_metrics(batch_size, batch_metrics: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        if not batch_metrics:
            return [{}] * batch_size 
        
        per_item_metrics = []
        
        metrics_numpy = {}
        for k, v in batch_metrics.items():
            v = v.detach().cpu().numpy()
            if v.ndim == 0:  # Scalar tensor
                metrics_numpy[k] = np.full(batch_size, float(v))
            else:  # Per-item tensor
                metrics_numpy[k] = v.reshape(batch_size)
        
        for i in range(batch_size):
            item_metrics = {
                k: float(v[i]) 
                for k, v in metrics_numpy.items()
            }
            per_item_metrics.append(item_metrics)
        
        return per_item_metrics 
    
