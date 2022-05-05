---
tags:
- conditional-image-generation
- image-to-image
- gan
- cyclegan

license: mit
---

# Image-to-image translation
A repository of image-to-image translation models developed for the Huggan hackaton [here](https://github.com/huggingface/community-events/tree/main/huggan).   

Currently it supports the cyclegan architecture for image-to-image translation


# Installation

Firstly, after cloning [this repository](https://github.com/chris1nexus/image2image.git), run   
```bash
cd image2image
pip install .
```

Then, set the wandb API_KEY if you wish to log all results to wandb with:   
```bash
wandb login API_KEY
```

If you plan on uploading the resulting models to an huggingface repository, make sure to also login with your huggingface API_KEY with the following command:   
```bash
huggingface-cli login 
```

Before starting the model training, it is necessary to configure the accelerate environment according to your available computing resource with the command:  
```bash
accelerate config
```

After this, everything is setup to start training the image2image models



# Training
To train an image2image translation model, one must only specify the dataset names   
following the huggingface dataset convention {organization_name}/{dataset_name} in the following command line args:   
* --source_dataset_name
* --target_dataset_name 

Single or multi gpu training is automatically set up by accelerate by running the following command to easily start the training of the image translation model


```bash
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
        train.py \
        --source_dataset_name Chris1/GTA5 \ 
        --target_dataset_name Chris1/cityscapes \        
        --batch_size 8 \
        --beta1 0.5 \
        --beta2 0.999 \
        --channels 3 \ 
        --checkpoint_interval 5 \
        --decay_epoch 80 \
        --epoch 0 \
        --image_size 256 \
        --lambda_cyc 10.0 \
        --lambda_id 5.0 \ 
        --lr 0.0002 \
        --mixed_precision no \
        --model_name cyclegan \
        --n_residual_blocks 9 \
        --num_epochs 200 \
        --num_workers 8 \
        --organization_name Chris1 \
        --push_to_hub \
        --sample_interval 10 \
        --wandb \
        --output_dir experiments
```
# TODO

### Provide a wider variety of architectures for the image2image translation task:     
    * [x] CycleGAN   
    * [ ] Cycada    
    * [ ] ...       

### Architecture modifications:  
    * [ ] FID loss CycleGAN  
    * [ ] ...     


# About


_Adapted by Christian Cancedda and inspired by Hugging Face with love_ ❤️# image2image
