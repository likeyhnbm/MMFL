#!/bin/bash 

python main.py --method "pretrain" --client_sample 0.25 --client_number 64 --thread_number 16 --comm_round 50 --lr 1e-5 --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_small_patch16_224 --optimizer sgd --sam_mode none --vpt_projection -1 
python main.py --method "prompt_official" --client_sample 0.25 --client_number 64 --thread_number 16 --comm_round 50 --lr 1e-3 --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_small_patch16_224 --optimizer sgd --sam_mode none --vpt_projection -1
python main.py --method "scratch" --client_sample 0.25 --client_number 64 --thread_number 16 --comm_round 50 --lr 1e-5 --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_small_patch16_224 --optimizer sgd --sam_mode none --vpt_projection -1 
