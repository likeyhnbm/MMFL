#!/bin/bash 


for optimizer in 'sgd' 'adamw'
do
    for lr in 1e-4 5e-4 1e-3 5e-3 1e-2
    do
        python main.py --method prompt --client_number 16 --thread_number 16 --comm_round 50 --lr $lr --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_small_patch16_224 --optimizer $optimizer
    done
done
