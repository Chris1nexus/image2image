---
tags:
- conditional-image-generation
- image-to-image
- gan
- cyclegan
# See a list of available tags here:
# https://github.com/huggingface/hub-docs/blob/main/js/src/lib/interfaces/Types.ts#L12
# task: unconditional-image-generation or conditional-image-generation or image-to-image
license: mit
---

# CycleGAN for unpaired image-to-image translation. 

## Model description  

CycleGAN for unpaired image-to-image translation.   
Given two image domains A and B, the following components are trained end2end to translate between such domains:   
- A generator A to B, named G_AB conditioned on an image from A   
- A generator B to A, named G_BA conditioned on an image from B   
- A domain classifier D_A, associated with G_AB   
- A domain classifier D_B, associated with G_BA    


At inference time, G_AB or G_BA are relevant to translate images, respectively A to B or  B to A.  
In the general setting, this technique provides style transfer functionalities between the selected image domains A and B.   
This allows to obtain a generated translation by G_AB, of an image from domain A that resembles the distribution of the images from domain B, and viceversa for the generator G_BA.  
Under these framework, these aspects have been used to perform style transfer between NFT collections.   
A collection is selected as domain A, another one as domain B and the CycleGAN provides forward and backward translation between A and B.   
This has showed to allows high quality translation even in absence of paired sample-ground-truth data.  
In particular, the model performs well with stationary backgrounds (no drastic texture changes in the appearance of backgrounds) as it is capable of recognizing the attributes of each of the elements of an NFT collections.  
An attribute can be a variation in type of dressed fashion items such as sunglasses, earrings, clothes and also face or body attributes with respect to a common template model of the given NFT collection).    


## Intended uses & limitations

#### How to use

```python
---
tags:
- conditional-image-generation
- image-to-image
- gan
- cyclegan
# See a list of available tags here:
# https://github.com/huggingface/hub-docs/blob/main/js/src/lib/interfaces/Types.ts#L12
# task: unconditional-image-generation or conditional-image-generation or image-to-image
license: mit
---

# CycleGAN for unpaired image-to-image translation. 

## Model description  

CycleGAN for unpaired image-to-image translation.   
Given two image domains A and B, the following components are trained end2end to translate between such domains:   
- A generator A to B, named G_AB conditioned on an image from A   
- A generator B to A, named G_BA conditioned on an image from B   
- A domain classifier D_A, associated with G_AB   
- A domain classifier D_B, associated with G_BA    


At inference time, G_AB or G_BA are relevant to translate images, respectively A to B or  B to A.  
In the general setting, this technique provides style transfer functionalities between the selected image domains A and B.   
This allows to obtain a generated translation by G_AB, of an image from domain A that resembles the distribution of the images from domain B, and viceversa for the generator G_BA.  
Under these framework, these aspects have been used to perform style transfer between synthetic data obtained from a simulated driving dataset, GTA5, and the real driving data from Cityscapes.   
This is of paramount importance to develop autonomous driving perception deep learning models, as this allows to generate synthetic data with automatic annotations which resembles real world images, without requiring the intervention of a human annotator.  
This is fundamental because a manual annotator has been shown to require 1.5 to 3.3 hours to create semantic and instance segmentation masks for a single images.  
These have been provided in the original [cityscapes paper (Cordts et al 2016)](https://arxiv.org/abs/2104.13395) and the [adverse condition dataset (Sakaridis et al. 2021)](https://arxiv.org/abs/2104.13395) paper.    

  
Hence the CycleGAN provides forward and backward translation between synthetic and real world data.   
This has showed to allows high quality translation even in absence of paired sample-ground-truth data.  
The idea behind such model is that as the synthetic data distribution gets closer to the real world one, deep models do not suffer from degraded performance due to the domain shift issue.   
A broad literature is available on the minimization of the domain shift, under the research branch of domain adaptation and transfer learning, of which image translation models provide an alternative approach


## Intended uses & limitations

#### How to use

```python
import os
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from im2im.models.architecture.cyclegan import GeneratorResNet
import torch.nn as nn
import torch
import gradio as gr
import glob




def pred_pipeline(img, transforms):
        orig_shape = img.shape
        input = transforms(img)
        input = input.unsqueeze(0)
        output = model(input)

        out_img = make_grid(output,#.detach().cpu(),
                           nrow=1, normalize=True)  
        out_transform = Compose([
            T.Resize(orig_shape[:2]),
            T.ToPILImage()
        ])
        return out_transform(out_img)




n_channels = 3
image_size = 512
input_shape = (image_size, image_size)

transform = Compose([
     T.ToPILImage(),
        T.Resize(input_shape),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


model = GeneratorResNet.from_pretrained('Chris1/sim2real', input_shape=(n_channels, image_size, image_size), 
                num_residual_blocks=9)
                
real_images = model(synthetic_images)      
```



#### Limitations and bias

Due to the absence of paired data, some background parts of the synthetic images are seldom wrongly translated, e.g. sky is translated to vegetation.   
Additional pretext tasks in parallel to the discriminative classifier of fake and real samples could improve the result.  
One easy improvement is the use of an additional parallel branch that performs semantic segmentation on the synthetic data, in order to learn features which are common to sky and vegetation, thus disentangling their representations as separate classes.    

## Training data


The CycleGAN model is trained on an unpaired dataset of samples from synthetic and real driving data, respectively from the GTA5 and Cityscapes datasets.   
To this end, the synthetic-to-real dataset can be loaded by means of the function load_dataset in the huggingface library, as follows.
```python
from datasets import load_dataset

unpaired_dataset = load_dataset("huggan/sim2real_gta5_to_cityscapes")

```
This dataset contains two columns, imageA and imageB representing respectively the GTA5 and Cityscapes data.  
Due to the fact that the two columns have to be of the same length, GTA5 is subsampled in order to reach the same number of samples provided by the Cityscapes train split (2975)


## Training procedure
#### Preprocessing
The following transformations are applied to each input sample of synthetic and real data.   
The input size is fixed to RGB images of height, width = 512, 512.
This choice has been made in order to limit the impact of upsampling the translated images to higher resolutions.
```python
n_channels = 3
image_size = 512
input_shape = (image_size, image_size)

transform = Compose([
    T.ToPILImage(),
    T.Resize(input_shape),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

#### Hardware  
The configuration has been tested on single GPU setup on a RTX5000 and A5000, as well as multi-gpu single-rank distributed setups composed of 2 of the mentioned GPUs.

#### Hyperparameters
The following configuration has been kept fixed for all translation models:   
- learning rate 0.0002   
- number of epochs 200
- learning rate decay activation at epoch 100
- number of residual blocks of the cyclegan 9
- image size 512x512
- number of channels=3
- cycle loss weight 10.0
- identity loss weight 5.0
- optimizer ADAM with beta1 0.5 and beta2 0.999
- batch size 8
- NO mixed precision training

## Eval results

#### Generated Images

In the provided images, row0 and row2 represent the synthetic and real images from the respective datasets.  
Row1 is the translation of the immediate above images in row0(synthetic) by means of the G_AB translation model, to the real world style.  
Row3 is the translation of the immediate above images in row2(real) by means of the G_BA translation model, to the synthetic world style.  

 Visualization over the training iterations for [synthetic (GTA5) to real (Cityscapes) translation](https://wandb.ai/chris1nexus/experiments_cyclegan_s2r_hp_opt--10/reports/CycleGAN-sim2real-training-results--VmlldzoxODUyNTk4?accessToken=tow3v4vp02aurzodedrdht15ig1cx69v5mited4dm8bgnup0z192wri0xtftaeqj) 


### References
```bibtex
@misc{https://doi.org/10.48550/arxiv.1703.10593,
  doi = {10.48550/ARXIV.1703.10593},
  
  url = {https://arxiv.org/abs/1703.10593},
  
  author = {Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  
  publisher = {arXiv},
  
  year = {2017},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

