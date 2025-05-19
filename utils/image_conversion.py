import torch
from PIL import Image
import numpy as np

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    np_array = tensor.cpu().float().numpy()
    is_batch = len(np_array.shape) == 4
    
    # Transpose batch or single image
    np_array = np.transpose(np_array, (0,2,3,1) if is_batch else (1,2,0))
    np_array = (np_array * 255).round().astype("uint8")
    
    return [Image.fromarray(img) for img in np_array] if is_batch else Image.fromarray(np_array)