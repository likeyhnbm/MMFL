    for dataset in dataset/resisc45/
    do
        # 16 1
        sh scripts/resisc_optimal/scratch.sh 16 1 homo 0.1 8 $dataset 
        sh scripts/resisc_optimal/pretrain.sh 16 1 hetero 0.1 8 $dataset 
        sh scripts/resisc_optimal/pretrain.sh 16 1 homo 0.1 8 $dataset 
        sh scripts/resisc_optimal/head.sh 16 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/head.sh 16 1 homo 0.1 16 $dataset 
        sh scripts/resisc_optimal/bias.sh 16 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/bias.sh 16 1 homo 0.1 16 $dataset 
        sh scripts/resisc_optimal/adapter.sh 16 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/adapter.sh 16 1 homo 0.1 16 $dataset 
        sh scripts/resisc_optimal/prompt.sh 16 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/prompt.sh 16 1 homo 0.1 16 $dataset 
        # 16 0.125
        sh scripts/resisc_optimal/pretrain.sh 16 0.125 hetero 0.1 2 $dataset 
        sh scripts/resisc_optimal/pretrain.sh 16 0.125 homo 0.1 2 $dataset 
        sh scripts/resisc_optimal/head.sh 16 0.125 hetero 0.1 2 $dataset 
        sh scripts/resisc_optimal/head.sh 16 0.125 homo 0.1 2 $dataset 
        sh scripts/resisc_optimal/bias.sh 16 0.125 hetero 0.1 2 $dataset 
        sh scripts/resisc_optimal/bias.sh 16 0.125 homo 0.1 2 $dataset 
        sh scripts/resisc_optimal/adapter.sh 16 0.125 hetero 0.1 2 $dataset 
        sh scripts/resisc_optimal/adapter.sh 16 0.125 homo 0.1 2 $dataset 
        sh scripts/resisc_optimal/prompt.sh 16 0.125 hetero 0.1 2 $dataset 
        sh scripts/resisc_optimal/prompt.sh 16 0.125 homo 0.1 2 $dataset 
        # 64 1
        sh scripts/resisc_optimal/pretrain.sh 64 1 hetero 0.1 8 $dataset 
        sh scripts/resisc_optimal/pretrain.sh 64 1 homo 0.1 8 $dataset 
        sh scripts/resisc_optimal/head.sh 64 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/head.sh 64 1 homo 0.1 16 $dataset 
        sh scripts/resisc_optimal/bias.sh 64 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/bias.sh 64 1 homo 0.1 16 $dataset 
        sh scripts/resisc_optimal/adapter.sh 64 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/adapter.sh 64 1 homo 0.1 16 $dataset 
        sh scripts/resisc_optimal/prompt.sh 64 1 hetero 0.1 16 $dataset 
        sh scripts/resisc_optimal/prompt.sh 64 1 homo 0.1 16 $dataset 
        # 64 0.125
        sh scripts/resisc_optimal/pretrain.sh 64 0.125 hetero 0.1 8 $dataset 
        sh scripts/resisc_optimal/pretrain.sh 64 0.125 homo 0.1 8 $dataset 
        sh scripts/resisc_optimal/head.sh 64 0.125 hetero 0.1 8 $dataset 
        sh scripts/resisc_optimal/head.sh 64 0.125 homo 0.1 8 $dataset 
        sh scripts/resisc_optimal/bias.sh 64 0.125 hetero 0.1 8 $dataset 
        sh scripts/resisc_optimal/bias.sh 64 0.125 homo 0.1 8 $dataset 
        sh scripts/resisc_optimal/adapter.sh 64 0.125 hetero 0.1 8 $dataset 
        sh scripts/resisc_optimal/adapter.sh 64 0.125 homo 0.1 8 $dataset 
        sh scripts/resisc_optimal/prompt.sh 64 0.125 hetero 0.1 8 $dataset 
        sh scripts/resisc_optimal/prompt.sh 64 0.125 homo 0.1 8 $dataset         
    done