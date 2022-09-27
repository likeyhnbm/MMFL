# client_num client_sample partition_method alpha thread_num dataset custom
sh scripts/optimal/prompt.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --comm_round=500
