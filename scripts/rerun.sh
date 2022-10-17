# client_num client_sample partition_method alpha thread_num dataset dp

for seed in 42 3407
do
    for dataset in dataset/cifar100/ 
    do
        # Reduce sample client for fine-tuning
        sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.1 8 $dataset\ --seed=$seed
        sh scripts/optimal/pretrain.sh 64 0.0625 hetero 0.1 4 $dataset\ --seed=$seed
        sh scripts/optimal/pretrain.sh 64 0.03125 hetero 0.1 2 $dataset\ --seed=$seed
        sh scripts/optimal/pretrain.sh 64 0.015625 hetero 0.1 1 $dataset\ --seed=$seed
        # shuffle net
        sh scripts/optimal/shufflenet.sh 64 0.125 hetero 0.1 8 $dataset\ --seed=$seed
        # FedPEFT
        sh scripts/optimal/bias.sh 64 0.125 hetero 0.1 8 $dataset\ --seed=$seed
        sh scripts/optimal/adapter.sh 64 0.125 hetero 0.1 8 $dataset\ --seed=$seed
        sh scripts/optimal/prompt.sh 64 0.125 hetero 0.1 8 $dataset\ --seed=$seed
        # Head tuning
        sh scripts/optimal/head.sh 64 0.125 hetero 0.1 8 $dataset\ --seed=$seed
        sh scripts/optimal/head.sh 64 1 hetero 0.1 8 $dataset\ --seed=$seed
    done
done
