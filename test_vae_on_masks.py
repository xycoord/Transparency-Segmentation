from dataset_configuration import prepare_dataset
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
import time

dataset_name = 'trans10k'
dataset_path = '/home/xycoord/models/Trans10k/'
train_batch_size = 1
dataloader_num_workers = 4

(train_loader, val_loader, test_loader), dataset_config_dict = prepare_dataset(
            data_name=dataset_name,
            dataset_path=dataset_path,
            batch_size=train_batch_size,
            test_batch=1,
            datathread=dataloader_num_workers,
            logger=None)

base_model_path = "stabilityai/stable-diffusion-3.5-large"
# base_model_path = "stabilityai/stable-diffusion-2"
device = "cuda"
dtype = torch.bfloat16

print("Loading VAE")
vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", use_safetensors=True, torch_dtype=dtype).to(device).requires_grad_(False)
vae.eval()
vae_image_processor = VaeImageProcessor(do_normalize=True)

print("Done Loading VAE")

def contrast_score(image):
    image_shifted = image - 0.5
    return torch.mean(0.5-torch.abs(image_shifted))

def quantize_tensor(tensor):
    return (tensor > 0.5).float()

def intersection_over_union(tensor1, tensor2):
    intersection = torch.sum((tensor1 * tensor2) > 0)
    union = torch.sum((tensor1 + tensor2) > 0)
    iou = intersection / union
    return iou

total_iou = 0.0
batches = len(train_loader)
loop_time = time.time()
for step, batch in enumerate(tqdm(train_loader)):
    if step > 20:
        break
    new_time = time.time()
    print("Time for loop: ", new_time - loop_time)
    loop_time = new_time

     # load image and mask 
    # image_data = batch[0]
    mask = batch[1]

    # ==== Reshape data for stable diffusion standards ====
    # mask is only a single channel so copy it across 3
    mask_single = mask.unsqueeze(1)
    mask_stacked = mask_single.repeat(1,3,1,1) # dim 0 is batch?
    mask_stacked = mask_stacked.float() # the dataset has it as a float
    mask_normalized= vae_image_processor.normalize(mask_stacked)

    # ==== Encode and Decode ====
    # encode
    encode_start = time.time()
    latents = vae.encode(mask_normalized.to('cuda').to(dtype)).latent_dist.sample()
    encode_end = time.time()
    print("Time to encode: ", encode_end - encode_start)
    continue
    
    # decode
    output = vae.decode(latents).sample
    
    output = output.cpu()
    output_tensor = vae_image_processor.postprocess(output, output_type='pt')
    output_quantized = quantize_tensor(output_tensor)
    # output_PIL = vae_image_processor.postprocess(output, output_type='pil')
    # output_PIL[0].show()
    iou = intersection_over_union(mask_stacked,output_quantized)
    total_iou += iou


print("Average IOU: ", total_iou/batches)