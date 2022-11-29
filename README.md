# Let's Unleash the Network Judgement: A Self-supervised Approach for Cloud Image Analysis

PyTorch implementation for Cloud Image Analysis using DINO (Cloud-DINO).

This project is based in the [original DINO](https://github.com/facebookresearch/dino) repository by [Facebook AI research group](https://ai.facebook.com/).

## Training Cloud-DINO on a node on 8 GPUs

We run it on an [ALCF](https://alcf.anl.gov/) cluster called [ThetaGPU](https://www.alcf.anl.gov/alcf-resources/theta).

To run it during 10 min do

`qsub -n 1 -q full-node -t 10 -A your_project ./train_Cloud_DINO.sh`

This is the `train_Cloud_DINO.sh` script for training in a node with 8 GPUs

```
#!/bin/sh

# Common paths
sky_images_path='/sky/images/path'
singularity_image_path='/path/to/the/singularity/container/your_singularity_image_file.sif'
cloud_dino_path='/path/to/cloud_dino'
train_cloud_dino_path='/path/to/cloud_dino/cloud_dino_training.py'
model_path='/path/to/the/model'

cd $cloud_dino_path
singularity exec --nv -B $sky_images_path:/Sky_Images $singularity_image_path python -m torch.distributed.launch --nproc_per_node=8 $train_cloud_dino_path --arch vit_small --data_path /Sky_Images --output_dir $model_path
```

## Inferencing using a Cloud-DINO trained model on 8 GPUs

To run it during 10 min do

`qsub -n 1 -q full-node -t 10 -A your_project ./inference_Cloud_DINO.sh`

This is the `inference_Cloud_DINO.sh` script for making inference on Cloud-DINO in a node with 8 GPUs

```
#!/bin/sh

# Common paths
sky_images_path='/sky/images/path'
singularity_image_path='/path/to/the/singularity/container/your_singularity_image_file.sif'
cloud_dino_path='/path/to/cloud_dino'
inference_cloud_dino_path='/path/to/cloud_dino/cloud_dino_inference.py'
model_path='/path/to/the/model'
output_path='/path/to/output/features'

cd $cloud_dino_path
singularity exec --nv -B $sky_images_path:/Sky_Images,$model_path:/Model,$output_path:/Output $singularity_image_path python -m torch.distributed.launch --nproc_per_node=8 $inference_cloud_dino_path --data_path /Sky_Images --pretrained_weights /Model/checkpoint0000.pth --dump_features /Output
```
