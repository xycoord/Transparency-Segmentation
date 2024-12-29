from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from pathlib import Path


base_model_path = "stabilityai/stable-diffusion-3.5-large"
device = "cuda"
dtype = torch.bfloat16
prompt = ""
save_folder_path = Path("./precomputed_prompt_embeddings")
prompt_embeds_path = save_folder_path / "empty_prompt_embeds.pt"
pooled_prompt_embeds_path = save_folder_path / "empty_pooled_prompt_embeds.pt"


# ======= Load Models =======

print("Loading Tokenizers and Encoders")
tokenizer_1 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer_2")
# t5_tokenizer = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_3")

text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(base_model_path, subfolder="text_encoder", use_safetensors=True, torch_dtype=dtype).to(device)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model_path, subfolder="text_encoder_2", use_safetensors=True, torch_dtype=dtype).to(device)
# t5_encoder = T5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder_3", use_safetensors=True).to(device)

print("Loading Transformer")
transformer = SD3Transformer2DModel.from_pretrained(
    base_model_path, 
    subfolder="transformer",
    use_safetensors=True,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
    ).to(device)


# ======= Setup Pipeline =======

print("Setting up pipeline")
prompt_embed_pipeline = StableDiffusion3Pipeline.from_pretrained(
    base_model_path,
    tokenizer=tokenizer_1,
    text_encoder=text_encoder_1,
    tokenizer_2=tokenizer_2,
    text_encoder_2=text_encoder_2,
    tokenizer_3=None,
    text_encoder_3=None,
    transformer=transformer,
    vae=None,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
prompt_embed_pipeline.enable_model_cpu_offload()

print("Encoding Prompts")
with torch.no_grad():
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = prompt_embed_pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt="",
        max_sequence_length=77,
    )

print("negative_prompt_embeds.shape:", negative_prompt_embeds.shape)
print("prompt_embeds.shape:", negative_prompt_embeds.shape)

prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
print("prompt_embeds.shape:", prompt_embeds.shape)
print("pooled_prompt_embeds.shape:", pooled_prompt_embeds.shape)

# print("Saving prompt embeddings.")
# save_folder_path.mkdir(exist_ok=True, parents=True)
# torch.save(prompt_embeds, prompt_embeds_path)
# torch.save(pooled_prompt_embeds, pooled_prompt_embeds_path)