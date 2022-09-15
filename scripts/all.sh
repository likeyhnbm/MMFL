# client_num client_sample partition_method alpha thread_num dataset dp
# sh scripts/custom/adapter.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --dp
sh scripts/custom/pretrain.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --dp
# sh scripts/custom/head.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --dp
# sh scripts/custom/prompt.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --dp
sh scripts/custom/bias.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --dp