#!/bin/bash 
for vit_type in vit_small_patch16_224 vit_base_patch16_224
do
        for lr in 0.1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 
        do
                python centralized.py --method "prompt_official" --client_sample 1 --client_number 1 --thread_number 1 --comm_round 50 --lr $lr --epochs 10 --data_dir dataset/cifar100/ --batch_size 2048 --vpt_type Deep --vit_type $vit_type --optimizer sgd --sam_mode none --vpt_projection -1
        done
done