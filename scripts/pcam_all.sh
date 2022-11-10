    for dataset in dataset/pcam/ 
    do
        # 16 1
        # sh scripts/optimal/scratch.sh 16 1 homo 0.1 8 $dataset --sample_num=20000
        sh scripts/optimal/scratch.sh 16 1 hetero 0.1 8 $dataset --sample_num=20000
        sh scripts/optimal/scratch.sh 16 1 hetero 0.1 8 dataset/resisc45
        # sh scripts/optimal/pretrain.sh 16 1 hetero 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/pretrain.sh 16 1 homo 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 16 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 16 1 homo 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 16 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 16 1 homo 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 16 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 16 1 homo 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 16 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 16 1 homo 0.5 16 $dataset --sample_num=20000
        # # 16 0.125
        # sh scripts/optimal/pretrain.sh 16 0.125 hetero 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/pretrain.sh 16 0.125 homo 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 16 0.125 hetero 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 16 0.125 homo 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 16 0.125 hetero 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 16 0.125 homo 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 16 0.125 hetero 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 16 0.125 homo 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 16 0.125 hetero 0.5 2 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 16 0.125 homo 0.5 2 $dataset --sample_num=20000
        # # 64 1
        # sh scripts/optimal/pretrain.sh 64 1 hetero 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/pretrain.sh 64 1 homo 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 64 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 64 1 homo 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 64 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 64 1 homo 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 64 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 64 1 homo 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 64 1 hetero 0.5 16 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 64 1 homo 0.5 16 $dataset --sample_num=20000
        # # 64 0.125
        # sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/pretrain.sh 64 0.125 homo 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/head.sh 64 0.125 homo 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/bias.sh 64 0.125 homo 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/adapter.sh 64 0.125 homo 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000
        # sh scripts/optimal/prompt.sh 64 0.125 homo 0.5 8 $dataset --sample_num=20000        
    done