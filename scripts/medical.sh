# client_num client_sample partition_method alpha thread_num
# sh scripts/custom/prompt.sh 64 0.125 homo 0.1 8 dataset/chestxray/
sh scripts/chest/prompt.sh 16 1 homo 0.1 8 dataset/chestxray/
sh scripts/chest/head.sh 16 1 homo 0.1 8 dataset/chestxray/
sh scripts/chest/adapter.sh 16 1 homo 0.1 8 dataset/chestxray/
# sh scripts/chest/bias.sh 16 1 homo 0.1 8 dataset/chestxray/
# sh scripts/chest/pretrain.sh 16 1 homo 0.1 4 dataset/chestxray/
