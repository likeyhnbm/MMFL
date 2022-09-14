# client_num client_sample partition_method alpha thread_num dataset dp
sh scripts/custom/bias.sh 16 1 hetero 0.1 16 dataset/cifar100/
sh scripts/custom/head.sh 16 1 hetero 0.1 16 dataset/cifar100/

sh scripts/custom/scratch.sh 16 1 homo 0.1 16 dataset/cifar100/
sh scripts/custom/head.sh 16 1 homo 0.1 16 dataset/cifar100/
sh scripts/custom/pretrain.sh 16 1 homo 0.1 16 dataset/cifar100/
sh scripts/custom/prompt.sh 16 1 homo 0.1 16 dataset/cifar100/
sh scripts/custom/adapter.sh 16 1 homo 0.1 16 dataset/cifar100/
sh scripts/custom/bias.sh 16 1 homo 0.1 16 dataset/cifar100/
