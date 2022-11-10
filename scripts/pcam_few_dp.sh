    for dataset in dataset/pcam/ 
    do
        for k in 2000 1500 1000
        # 64 0.125
        do
            sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=$k
            sh scripts/optimal/head.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=$k
            sh scripts/optimal/bias.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=$k
            sh scripts/optimal/adapter.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=$k
            sh scripts/optimal/prompt.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=$k
        done
        sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --dp
        sh scripts/optimal/head.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --dp
        sh scripts/optimal/bias.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --dp
        sh scripts/optimal/adapter.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --dp
        sh scripts/optimal/prompt.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --dp

    done