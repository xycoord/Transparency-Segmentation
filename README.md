# Transparency Segmentation (SD3.5)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)](https://pytorch.org/)

Training and inference scripts for a Transparency Segmentation model fine-tuned from Stable Diffusion 3.5 with the Trans10k Dataset.

## End-to-End Fine Tuning

This branch implements the 1-step end-to-end fine tuning method introduced in [Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://arxiv.org/pdf/2409.11355). This is deterministic, more efficient, and yields better results than the marigold inspired diffusion version I implemented on the [diffusion-ft](https://github.com/xycoord/Transparency-Segmentation/tree/diffusion-ft) branch.

## Results

This model performs on par with SOTA on the TransLab dataset.

| Model | Difficulty | IOU    | Recall | MSE    |
|-------|------------|--------|--------|--------|
| TransLab  | Mix    | 0.8763 |        |        |
| This work |        | 0.8819 | 0.9456 | 0.0312 |
| TransLab  | Easy   | 0.9223 |        |        |
| This work |        | 0.9214 | 0.9675 | 0.0189 |
| TransLab  | Hard   | 0.721  |        |        |
| This work |        | 0.7346 | 0.8641 | 0.0772 |

*Note: the IOU for Translab is mIOU since it separates Things and Stuff into separate segmentation classes. Therefore the metrics aren't exactly comparable*

### Example Predictions

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|
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

This repo assumes use of Stable Diffusion 3 or 3.5 as the pre-trained model. I specifically use **3.5 medium**. A previous version used Stable Diffusion 2. Notable changes include:
- Predictions made at 1024 x 1024px with time step sampling optimized for this resolution during training
- 16 channel latent space (vs 4 in SD2). This [has been shown](https://arxiv.org/pdf/2309.15807) to improve preservation of fine details leading to better reproduction of text. I am yet to conclude whether it improves reproduction of useful cues for transparent objects.

## Comparison with Diffusion Approach

The end-to-end approach outperforms my Marigold-inspired diffusion implementation in both speed and accuracy. It is much faster (single forward pass vs. ~100 denoising steps) and more accurate.

The diffusion model's primary weakness is **transparency classification** - determining whether objects should be segmented as transparent or not. When it correctly identifies an object as transparent, it produces high-quality segmentations with accurate edges. However, it frequently misclassifies transparent objects as opaque (or vice versa).

This might occur because transparency classification is determined by low-frequency content early in the denoising process. During training, only a small fraction of examples are of sufficiently early timesteps to teach the model this binary classification decision. By contrast, the end-to-end approach learns transparency classification from every training example.


## Limitations and Failure Cases

**Stickers on glass**: Consistently confused by opaque elements (stickers, labels) on transparent surfaces.

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2165_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2165_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2165_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2165_img.png?raw=true">


**Reflections**: Frequently produces false positives on reflections or images of transparent objects.

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/50_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/50_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/50_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/50_img.png?raw=true">


**Ambiguous transparency**: Struggles with boundary cases like opaque objects behind glass or items in transparent packaging. The model often classifies these as transparent whilst the dataset does not.

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/322_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/322_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/322_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/322_img.png?raw=true">
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/915_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/915_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/915_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/915_img.png?raw=true">

**Low resolution confusion**: May misclassify transparency when fine details are only visible at high resolution, perhaps due to VAE limitations.

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/160_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/160_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/160_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/160_img.png?raw=true">
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/256_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/256_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/256_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/256_img.png?raw=true">
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2085_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2085_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2085_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2085_img.png?raw=true">


**Ambiguous cases**: Some test examples are ambiguous even when at full resolution. The model will often disagree with the test example in these cases but it is unclear which is correct.

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2138_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2138_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2138_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/2138_img.png?raw=true">


**Dataset errors**: Some test examples appear to have incorrect ground truth annotations, though these remain uncorrected for benchmark consistency.

| Prediction | Raw Pred | Ground Truth | Image |
|-|-|-|-|
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/730_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/730_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/730_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/730_img.png?raw=true">
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/777_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/777_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/777_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/777_img.png?raw=true">
|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/1180_maskq.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/1180_pred.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/1180_maskgt.png?raw=true">|<img width="200" src="https://github.com/xycoord/Transparency-Estimation/blob/main/sample_results/failure_cases/1180_img.png?raw=true">

**Fuzzy predictions**: The raw predictions often have soft, blurry edges whilst the ground truth has sharp boundaries. When quantised to binary masks, this results in boundary noise with serrated, non-contiguous edges.



## Setup 

### Dataset
I use the Trans10k dataset from [Segmenting Transparent Objects in the Wild](https://arxiv.org/abs/2003.13948). It consists of pairs of images and transparency segmentation masks.

Google Drive links to download the data can be found on [the paper's website](https://xieenze.github.io/projects/TransLAB/TransLAB.html). The dataloader in this repo is based on [the original](https://github.com/xieenze/Segment_Transparent_Objects) but is much simplified for my purposes.

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
It's used to load the pre-trained models from Hugging Face Hub.


## Tracking
I use Weights and Biases for experiment tracking. This is handled by Accelerate.
To learn how to set this up or swap trackers see [this guide](https://huggingface.co/docs/accelerate/en/usage_guides/tracking).

## License
This project is licensed under the MIT License - see the LICENSE file for details.