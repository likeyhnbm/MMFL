# client_num client_sample partition_method alpha thread_num dataset dp
sh scripts/custom/adapter.sh 64 0.125 homo 0.5 8 dataset/cifar100/
sh scripts/custom/bias.sh 64 0.125 homo 0.5 8 dataset/cifar100/
sh scripts/custom/head.sh 64 0.125 homo 0.5 8 dataset/cifar100/
sh scripts/custom/prompt.sh 64 0.125 homo 0.5 8 dataset/cifar100/