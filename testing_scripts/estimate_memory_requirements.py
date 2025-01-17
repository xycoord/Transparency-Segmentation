from diffusers import SD3Transformer2DModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

# Load model of interest
model = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="transformer")
# Estimate memory requirements for use with DeepSpeed
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)