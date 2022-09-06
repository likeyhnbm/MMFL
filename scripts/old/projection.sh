#!/bin/bash 

for projection in -1 512 256 1024
do
    python main.py --method prompt_official --client_number 16 --thread_number 16 --comm_round 50 --lr 1e-3 --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_small_patch16_224 --optimizer sgd --sam_mode none --vpt_projection $projection 
done