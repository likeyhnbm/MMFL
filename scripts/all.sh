# client_num client_sample partition_method alpha thread_num dataset dp
<<<<<<< HEAD
sh scripts/custom/prompt.sh 16 1 homo 0.1 8 dataset/cifar100/
sh scripts/custom/adapter.sh 16 1 homo 0.1 8 dataset/cifar100/
sh scripts/custom/bias.sh 16 1 homo 0.1 8 dataset/cifar100/
=======
sh scripts/custom/pretrain.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --dp
sh scripts/custom/bias.sh 64 0.125 hetero 0.1 8 dataset/cifar100/ --dp
>>>>>>> 42dd119bb820a0a18febed91e861cad6b08d0123
