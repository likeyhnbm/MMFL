# client_num client_sample partition_method alpha thread_num
sh scripts/custom/prompt.sh 64 0.125 hetero 0.1 8 dataset/chestxray/
sh scripts/custom/pretrain.sh 64 0.125 hetero 0.1 4 dataset/chestxray/
