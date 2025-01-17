# "Marigold" Transparency Segmentation v2 (SD3.5)

Training scripts for a Marigold inspired Transparency Segmentation model using Stable Diffusion 3.5 and the Trans10k Dataset.

## Method

### *Inspiration from* Marigold  [[Web Page]](https://marigoldmonodepth.github.io/) [[Paper]](https://arxiv.org/abs/2312.02145)

The idea I take from Marigold is to use pre-trained generative diffusion models as a starting point for image-to-image vision tasks. Marigold does this for monocular depth estimation and I use the same technique for transparent object image segmentation. The hope is that a) the model doesn't have to learn key vision concepts from scratch and b) will generalise well outside the training distribution.

During training, both the image and mask are encoded into the latent space. The mask is then noised as per the noise schedule. The latents are concatenated along the channel dimension such that the transformer is conditioned both with the noisy latent (as in standard diffusion models) as well as the image latent. Since for inference, we begin with noise, the whole diffusion pipeline is conditioned on the image and generates a prediction of the mask.

Since the process is stochastic, I ensemble multiple predictions with a mean and quantize the result to {0, 1} effectively having the predictions vote on the classification of each pixel.

### Stable Diffusion 3.5 [[Paper]](https://arxiv.org/abs/2403.03206)

This repo assumes we're using Stable Diffusion 3 or 3.5 as the pre-trained model.
The previous version used Stable Diffusion 2 as in Marigold. Notable changes include:
- Rectified Flow ensures a 0 SNR at t=1 so multi-resolution noise is redundant and has been removed
- Predictions made at 1024 x 1024px with time step sampling optimized for this resolution during training
- 16 channel latent space (vs 4 in SD2). This [has been shown](https://arxiv.org/pdf/2309.15807) to improve preservation of fine details leading to better reproduction of text. I am yet to conclude whether it improves reproduction of useful cues for transparent objects.


## Setup 

### Dataset
I use the Trans10k dataset from [Segmenting Transparent Objects in the Wild](https://arxiv.org/abs/2003.13948).
It consists of pairs of images and transparency segmentation masks.
Google Drive links to download the data can be found on [the paper's website](https://xieenze.github.io/projects/TransLAB/TransLAB.html).
The dataloader in this repo is based on [the original](https://github.com/xieenze/Segment_Transparent_Objects) but is much simplified for our purposes.

### Dependencies
Tested using python 3.9 and 3.10. Newer versions <=3.12 will probably work.

To install the dependencies into an active virtual environment:

```
pip install -r requirements.txt
```

*Note: Deepspeed 0.16 and Transformers create [this bug](https://github.com/microsoft/DeepSpeed/issues/6793) so use Deepseed 0.15.4 for now.*

### Environment Variables
Examples given for Linux where the lines should be added to ~/.bashrc

To specify where the pretrained models are cached:
```
export HF_HOME=/path/to/cache
```

To speed up downloading of models (strongly recomended):
```
export HF_HUB_ENABLE_HF_TRANSFER=1
``` 

### Config
Arguments for the scripts are defined in `config.yaml` and loaded into `args`.
Any argument can be set either by editing `config.yaml` or using a command line argument when launching the script to override the default.

Make sure to set `output_dir` and `dataset_path` for your setup.

### System Requirements

I have successfully run training on:
- 4x Nvidia L4 (Total 96GB VRAM)
- ~100GB RAM

And Inference on:
- 1x Nvidia L4 (24GB VRAM)
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
We primarily use it to load the pre-trained models from Hugging Face Hub.


## Tracking
I use Weights and Biases for experiment tracking. This is handled by Accelerate.
To learn how to set this up or swap trackers see [this guide](https://huggingface.co/docs/accelerate/en/usage_guides/tracking).


## Current Progress/Results
This project is ongoing. While the code works and the model learns, I'm not convinced I've yet found the optimal hyper-parameters and learning rate schedule.

Here are some random samples from a recent validation run. As you can see some are almost perfect while others are far off. Which are good changes randomly between checkpoints.

| Prediction | Ground Truth | Image | Prediction | Ground Truth | Image |
|-|-|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_0.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_gt_0.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_img_0.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_1.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_gt_1.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_img_1.png?raw=true">|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_2.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_gt_2.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_img_2.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_3.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_gt_3.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_0_0_img_3.png?raw=true">|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_0.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_gt_0.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_img_0.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_1.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_gt_1.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_img_1.png?raw=true">|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_2.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_gt_2.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_img_2.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_3.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_gt_3.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_1_0_img_3.png?raw=true">|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_0.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_gt_0.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_img_0.png?raw=true"> |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_1.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_gt_1.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_img_1.png?raw=true">|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_2.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_gt_2.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_img_2.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_3.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_gt_3.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_2_0_img_3.png?raw=true">|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_0.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_gt_0.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_img_0.png?raw=true"> |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_1.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_gt_1.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_img_1.png?raw=true">|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_2.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_gt_2.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_img_2.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_3.png?raw=true">  |<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_gt_3.png?raw=true"> | <img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/val_3_0_img_3.png?raw=true">|