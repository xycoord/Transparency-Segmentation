from dataset_configuration import normalize_mask, prepare_dataset

dataset_name = 'trans10k'
dataset_path = '/home/xycoord/models/Trans10k/'
train_batch_size = 4
dataloader_num_workers = 4

(train_loader, val_loader, test_loader), dataset_config_dict = prepare_dataset(
            data_name=dataset_name,
            dataset_path=dataset_path,
            batch_size=train_batch_size,
            test_batch=1,
            datathread=dataloader_num_workers,
            logger=None)

for step, batch in enumerate(train_loader):
     # load image and mask 
    image_data = batch[0]
    mask = batch[1]

    # ==== Reshape data for stable diffusion standards ====
    # mask is only a single channel so copy it across 3
    mask_single = mask.unsqueeze(1)
    mask_stacked = mask_single.repeat(1,3,1,1) # dim 0 is batch?
    mask_stacked = mask_stacked.float() # the dataset has it as a float
    mask_normalized= normalize_mask(mask_stacked)

    break
