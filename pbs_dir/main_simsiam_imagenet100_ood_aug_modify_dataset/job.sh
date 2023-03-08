#!/bin/bash
#source /users/visics/ajha/.bash_custom

echo "Current path is $PATH"
echo "Running"
nvidia-smi
echo $CUDA_VISIBLE_DEVICES


source $VSC_HOME/.bash_custom

cd $VSC_DATA/ajha/code/visual_representation_learning/simsiam_out_of_distribution_augmentation_refinement/imagenet100


python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM main_simsiam_imagenet100_ood_aug_modify_dataset.py -a resnet50 --world-size 1 --batch-size 8 --rank 0 --fix-pred-lr /scratch/leuven/346/vsc34686/datasets/Imagenet_downloads/