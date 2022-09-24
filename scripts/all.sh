# client_num client_sample partition_method alpha thread_num dataset dp
sh scripts/custom/pretrain.sh 16 0.125 homo 0.1 2 dataset/cifar100/ 
sh scripts/custom/prompt.sh 16 0.125 homo 0.1 2 dataset/cifar100/
sh scripts/custom/adapter.sh 16 0.125 homo 0.1 2 dataset/cifar100/ 
sh scripts/custom/bias.sh 16 0.125 homo 0.1 2 dataset/cifar100/ 
sh scripts/custom/head.sh 16 0.125 homo 0.1 2 dataset/cifar100/
