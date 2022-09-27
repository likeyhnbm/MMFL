#!/bin/bash 
client_num=$1
client_sample=$2
partition=$3
alpha=$4
thread_number=$5
data_dir=$6
other=$7

for lr in 5e-3
do
        python main.py --partition_alpha $alpha --method "adapter" --client_sample $client_sample --client_number $client_num --thread_number $thread_number --lr $lr --data_dir $data_dir --batch_size 64 --vpt_type Deep --vit_type vit_base_patch16_224_in21k --optimizer sgd --sam_mode none --vpt_projection -1 --partition_method $partition $other
done
