# client_num client_sample partition_method alpha thread_num dataset dp
sh scripts/custom/prompt.sh 16 1 homo 0.1 8 dataset/cifar100/
sh scripts/custom/adapter.sh 16 1 homo 0.1 8 dataset/cifar100/
sh scripts/custom/bias.sh 16 1 homo 0.1 8 dataset/cifar100/
