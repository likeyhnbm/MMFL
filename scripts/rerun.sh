# client_num client_sample partition_method alpha thread_num dataset dp

for redo in 1 2
do
    for dataset in dataset/cifar100/ dataset/pcam/ 
    do
        # Reduce sample client for fine-tuning
        # sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.1 4 $dataset
        sh scripts/optimal/pretrain.sh 64 0.0625 hetero 0.1 4 $dataset
        sh scripts/optimal/pretrain.sh 64 0.03125 hetero 0.1 2 $dataset
        sh scripts/optimal/pretrain.sh 64 0.015625 hetero 0.1 1 $dataset
        # shuffle net
        sh scripts/optimal/shufflenet.sh 64 0.125 hetero 0.1 8 $dataset
        # FedPEFT
        sh scripts/optimal/bias.sh 64 0.125 hetero 0.1 8 $dataset
        sh scripts/optimal/adapter.sh 64 0.125 hetero 0.1 8 $dataset
        sh scripts/optimal/prompt.sh 64 0.125 hetero 0.1 8 $dataset
        # Head tuning
        sh scripts/optimal/head.sh 64 0.125 hetero 0.1 8 $dataset
        sh scripts/optimal/head.sh 64 1 hetero 0.1 8 $dataset
    done
done
