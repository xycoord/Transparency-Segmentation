from dataset_configuration import get_trans10k_train_loader, get_trans10k_test_loader, get_trans10k_val_loader
from utils import load_prompt_embeds

"""
This script is used to test the dataloaders for the Trans10k dataset and the Embeddings for the prompts.
It also serves as a place to develop data processing code e.g. reshaping without running the full training script which is time consuming.
"""

# Testing Parameters
device = 'cuda'
dataset_name = 'trans10k'
dataset_path = '/home/xycoord/models/Trans10k/'
train_batch_size = 4
dataloader_num_workers = 4

# Checking Empty Prompt Embeds
prompt_embeds, pooled_prompt_embeds = load_prompt_embeds('./precomputed_prompt_embeddings/')
prompt_embeds = prompt_embeds.to(device)
pooled_prompt_embeds = pooled_prompt_embeds.to(device)

print("Prompt Embeds Testing")
prompt_embeds = prompt_embeds.repeat(train_batch_size, 1, 1)
print("Prompt Embeds Shape: ", prompt_embeds.shape)
pooled_prompt_embeds = pooled_prompt_embeds.repeat(train_batch_size, 1 )
print("Pooled Prompt Embeds Shape: ", pooled_prompt_embeds.shape)

# Checking Data Loaders
train_loader = get_trans10k_train_loader(dataset_path, batch_size=train_batch_size)
val_loader = get_trans10k_val_loader(dataset_path, difficulty='easy')
test_loader = get_trans10k_test_loader(dataset_path, difficulty='hard')

def check_loader(loader):
     print("Loader Length: ", len(loader))
     for step, batch in enumerate(loader):

         image_data, mask, names = batch

         print("Image Data Shape: ", image_data.shape)
         print("Mask Shape: ", mask.shape)
         print("Names: ", names)

         break

print("Train Loader Testing")
check_loader(train_loader)

print("Val Loader Testing")
check_loader(val_loader)

print("Test Loader Testing")
check_loader(test_loader)