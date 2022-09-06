#!/bin/bash 

for prompt_num in 5 10 20 50 100
do
    for lr in 1e-1 5e-3
    do
        python main.py --partition_alpha 0.1 --method prompt_official --client_number 64 --thread_number 8 --client_sample 0.125 --comm_round 50 --lr $lr --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_base_patch16_224_in21k --optimizer sgd --sam_mode none --prompt_num $prompt_num
    done
done