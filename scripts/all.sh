# client_num client_sample partition_method alpha thread_num dataset dp
# sh scripts/chest/prompt.sh 64 0.125 homo 0.1 8 dataset/chestxray/ 
# sh scripts/chest/pretrain.sh 64 0.125 homo 0.1 4 dataset/chestxray/ 
# # sh scripts/chest/scratch.sh 64 0.125 homo 0.1 4 dataset/chestxray/ 
# # sh scripts/chest/adapter.sh 64 0.125 homo 0.1 8 dataset/chestxray/ 
# # sh scripts/chest/bias.sh 64 0.125 homo 0.1 8 dataset/chestxray/ 
# sh scripts/chest/head.sh 64 0.125 homo 0.1 8 dataset/chestxray/ 
# # sh scripts/chest/prompt.sh 64 0.125 homo 0.1 8 dataset/chestxray/ 

# hetero
sh scripts/dp/pretrain.sh 16 0.125 hetero 0.1 2 dataset/cifar100/
sh scripts/dp/prompt.sh 16 0.125 hetero 0.1 2 dataset/cifar100/
sh scripts/dp/adapter.sh 16 0.125 hetero 0.1 2 dataset/cifar100/ 
sh scripts/dp/bias.sh 16 0.125 hetero 0.1 2 dataset/cifar100/ 
sh scripts/dp/head.sh 16 0.125 hetero 0.1 2 dataset/cifar100/


