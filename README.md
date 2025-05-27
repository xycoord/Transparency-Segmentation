# Transparency Segmentation (SD3.5)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)](https://pytorch.org/)

Training and inference scripts for a Transparency Segmentation model fine-tuned from Stable Diffusion 3.5 with the Trans10k Dataset.

## End-to-End Fine Tuning

This branch implements the 1-step end-to-end fine tuning method introducted in [Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://arxiv.org/pdf/2409.11355). This is deterministic, more efficient, and yields better results than the marigold inspired diffusion version I implemented on the [diffusion-ft](https://github.com/xycoord/Transparency-Segmentation/tree/diffusion-ft) branch.

## Results

This model performs on par with SOTA on the TransLab dataset.

| Model | Difficulty | IOU    | Recall | MSE    |
|-------|------------|--------|--------|--------|
| TransLab | Mix     | 0.8763 |        |        |
| Ours     |         | 0.8819 | 0.9456 | 0.0312 |
| TransLab | Easy    | 0.9223 |        |        |
| Ours     |         | 0.9214 | 0.9675 | 0.0189 |
| TransLab | Hard    | 0.721  |        |        |
| Ours     |         | 0.7346 | 0.8641 | 0.0772 |

*Note: the IOU for Translab is mIOU since it separates Things and Stuff into separate segmenation classes. Therefore the metrics aren't exactly comparable*

### Example Predictions

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/1_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/1_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/1_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/1_img.png?raw=true">  
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/2_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/2_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/2_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/2_img.png?raw=true">  
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/4_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/4_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/4_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/4_img.png?raw=true">  
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/5_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/5_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/5_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/5_img.png?raw=true">  
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/6_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/6_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/6_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/6_img.png?raw=true">  
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/7_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/7_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/7_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/7_img.png?raw=true">  
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/11_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/11_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/11_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/11_img.png?raw=true">  




## Method

The core idea is to use pre-trained generative diffusion models as a starting point for image-to-image vision tasks. Unlike Marigold, which finetunes to a diffusion model, I use the vision transformer weights from stable diffusion as the starting point for an end-to-end model. 

During training, both the image and mask are encoded into the latent space using the frozen VAE. The transformer takes the image latent as input and outputs a prediction of the mask latent. The transformer is trained with an MSE loss with the known mask latent.

For inference, the input image is encoded. The transformer output (mask latent prediction) is decoded to give the result.

### Stable Diffusion 3.5 [[Paper]](https://arxiv.org/abs/2403.03206)

This repo assumes we're using Stable Diffusion 3 or 3.5 as the pre-trained model. I specifically use **3.5 medium**. A previous version used Stable Diffusion 2. Notable changes include:
- Predictions made at 1024 x 1024px with time step sampling optimized for this resolution during training
- 16 channel latent space (vs 4 in SD2). This [has been shown](https://arxiv.org/pdf/2309.15807) to improve preservation of fine details leading to better reproduction of text. I am yet to conclude whether it improves reproduction of useful cues for transparent objects.


## Setup 

### Dataset
I use the Trans10k dataset from [Segmenting Transparent Objects in the Wild](https://arxiv.org/abs/2003.13948). It consists of pairs of images and transparency segmentation masks.

Google Drive links to download the data can be found on [the paper's website](https://xieenze.github.io/projects/TransLAB/TransLAB.html). The dataloader in this repo is based on [the original](https://github.com/xieenze/Segment_Transparent_Objects) but is much simplified for our purposes.

I found a few of the pairs were at 90° rotation from each other. In my dataloader, pairs with different dimensions will throw an error. I fixed these examples manually, creating an updated version of the dataset.

### Dependencies
Tested using python 3.11 and pytorch 2.8.

Inference requires `attr` for faster image saving. Install with `apt install attr` or similar.

To install the dependencies into an active virtual environment or docker container:

```
pip install -r requirements.txt
```

*Note: Deepspeed 0.16 and Transformers create [this bug](https://github.com/microsoft/DeepSpeed/issues/6793) so use Deepseed 0.15.4 for now.*

#### Hugging Face
You will need a Hugging Face account and model access to download the stable diffusion weights. After setting this up and installing the python dependencies, run:
```
huggingface-cli login
```
and provide your access token.

### Environment Variables
Examples given for Linux where the lines should be added to ~/.bashrc

To specify where the pretrained models are cached:
```
export HF_HOME=/path/to/cache
```

To speed up downloading of models (strongly recommended):
```
export HF_HUB_ENABLE_HF_TRANSFER=1
```

To ensure multi-gpu setups work properly set the P2P mode e.g:
```
export NCCL_P2P_LEVEL=NVL
```
for NVLink.

### Config
Arguments for the scripts are defined in `config.yaml` and loaded into `args`.
Any argument can be set either by editing `config.yaml` or using a command line argument when launching the script to override the default.

Make sure to set `output_dir` and `dataset_path` for your setup.

### System Requirements

I have successfully run training on:
- 4x Nvidia L4 (Total 96GB VRAM)
- ~100GB RAM
Or
- 4x Nvidia A40 (Total 192GB VRAM)

And Inference on:
- 1x L4 (24GB) or 1x A40 (48GB)
- 48GB RAM


## Run

To run the training script:
```
accelerate launch --config_file accelerate_config.yaml trainer.py
```

To run inference:
```
accelerate launch --config_file accelerate_config.yaml val_inference.py
```

Other scripts don't use Accelerate and can be launched with python as normal.


## Key Libraries

### Accelerate (🤗 Hugging Face) [[Docs]](https://huggingface.co/docs/accelerate/en/index)
Accelerate handles distributed training across multiple GPUs.
Each GPU has it's own process which all run the training script so accelerate allows the processes to share information (e.g. gradients) and ensures that code which should only be run on a single process does.
Therefore it also handles optimizer steps, gradient accumulation, logging, and tracking.

Before running anything, update `accelerate_config.yaml` such that `num_processes` matches your number of GPUs *or* use your own config file.

### DeepSpeed [[Docs]](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
DeepSpeed helps with reducing memory usage in two ways:
1. Split Optimizer, Gradient and/or Parameters across GPUs
2. Offload Memory from VRAM to RAM or Disk (NVMe)

This project uses the Accelerate DeepSpeed Plugin which is set-up automatically as specified in `accelerate_config.yaml`. For details about these settings see the Docs.

*Note: When used, DeepSpeed takes over gradient clipping from Accelerate so make sure to set it in the DeepSpeed config. If using a single large GPU (e.g. 80GB A100), it may be worth disabling DeepSpeed. If you do this, add gradient clipping to the code with Accelerate.*


### Diffusers (🤗 Hugging Face) [[Docs]](https://huggingface.co/docs/diffusers/index)
Diffusers is a toolkit for working with diffusion model. 
We use it to load the pre-trained models from Hugging Face Hub.


## Tracking
I use Weights and Biases for experiment tracking. This is handled by Accelerate.
To learn how to set this up or swap trackers see [this guide](https://huggingface.co/docs/accelerate/en/usage_guides/tracking).
