#!/bin/bash 
for e in 20 50
do
        for lr in 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 0.1
        do
                python main.py --method "pretrain" --client_sample 0.125 --client_number 64 --thread_number 8 --comm_round 50 --lr $lr --epochs $e --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_base_patch16_224 --optimizer sgd --sam_mode none --vpt_projection -1 
                python main.py --method "prompt_official" --client_sample 0.125 --client_number 64 --thread_number 8 --comm_round 50 --lr $lr --epochs $e --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_base_patch16_224 --optimizer sgd --sam_mode none --vpt_projection -1
        done
done