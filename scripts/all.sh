for seed in 42 3407
do
    for dataset in dataset/cifar100/
    do
        # 16 1
        sh scripts/optimal/scratch.sh 16 1 homo 0.1 8 $dataset --seed=$seed
        sh scripts/optimal/pretrain.sh 16 1 hetero 0.1 8 $dataset --seed=$seed
        sh scripts/optimal/pretrain.sh 16 1 homo 0.1 8 $dataset --seed=$seed
        sh scripts/optimal/head.sh 16 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/head.sh 16 1 homo 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/bias.sh 16 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/bias.sh 16 1 homo 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/adapter.sh 16 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/adapter.sh 16 1 homo 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/prompt.sh 16 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/prompt.sh 16 1 homo 0.1 16 $dataset --seed=$seed
        # # 16 0.125
        # sh scripts/optimal/pretrain.sh 16 0.125 hetero 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/pretrain.sh 16 0.125 homo 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/head.sh 16 0.125 hetero 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/head.sh 16 0.125 homo 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/bias.sh 16 0.125 hetero 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/bias.sh 16 0.125 homo 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/adapter.sh 16 0.125 hetero 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/adapter.sh 16 0.125 homo 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/prompt.sh 16 0.125 hetero 0.1 2 $dataset --seed=$seed
        # sh scripts/optimal/prompt.sh 16 0.125 homo 0.1 2 $dataset --seed=$seed
        # 64 1
        sh scripts/optimal/pretrain.sh 64 1 hetero 0.1 8 $dataset --seed=$seed
        sh scripts/optimal/pretrain.sh 64 1 homo 0.1 8 $dataset --seed=$seed
        sh scripts/optimal/head.sh 64 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/head.sh 64 1 homo 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/bias.sh 64 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/bias.sh 64 1 homo 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/adapter.sh 64 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/adapter.sh 64 1 homo 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/prompt.sh 64 1 hetero 0.1 16 $dataset --seed=$seed
        sh scripts/optimal/prompt.sh 64 1 homo 0.1 16 $dataset --seed=$seed
        # 64 0.125
        # sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.1 8 $dataset 
        # sh scripts/optimal/pretrain.sh 64 0.125 homo 0.1 8 $dataset 
        # sh scripts/optimal/head.sh 64 0.125 hetero 0.1 8 $dataset 
        # sh scripts/optimal/head.sh 64 0.125 homo 0.1 8 $dataset 
        # sh scripts/optimal/bias.sh 64 0.125 hetero 0.1 8 $dataset 
        # sh scripts/optimal/bias.sh 64 0.125 homo 0.1 8 $dataset 
        # sh scripts/optimal/adapter.sh 64 0.125 hetero 0.1 8 $dataset 
        # sh scripts/optimal/adapter.sh 64 0.125 homo 0.1 8 $dataset 
        # sh scripts/optimal/prompt.sh 64 0.125 hetero 0.1 8 $dataset 
        # sh scripts/optimal/prompt.sh 64 0.125 homo 0.1 8 $dataset         
    done
done