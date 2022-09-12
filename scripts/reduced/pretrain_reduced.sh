#!/bin/bash 
for sample_num in 1000 5000 10000
do
        for lr in 0.001 0.0005 1e-5 5e-5 1e-4 5e-3
        do
                python main.py --partition_alpha 0.1 --method "pretrain" --client_sample 0.125 --client_number 64 --thread_number 8 --comm_round 50 --lr $lr --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_base_patch16_224_in21k --optimizer sgd --sam_mode none --vpt_projection -1 --sample_num $sample_num 
        done
done