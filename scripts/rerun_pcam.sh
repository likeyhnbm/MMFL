for seed in 42 3407
do
    for dataset in dataset/pcam/ 
    do
        # Reduce sample client for fine-tuning
        sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --seed=$seed
        sh scripts/optimal/pretrain.sh 64 0.0625 hetero 0.5 4 $dataset --sample_num=20000\ --seed=$seed
        sh scripts/optimal/pretrain.sh 64 0.03125 hetero 0.5 2 $dataset --sample_num=20000\ --seed=$seed
        sh scripts/optimal/pretrain.sh 64 0.015625 hetero 0.5 1 $dataset --sample_num=20000\ --seed=$seed
        # shuffle net
        sh scripts/optimal/shufflenet.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --seed=$seed
        # FedPEFT
        sh scripts/optimal/bias.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --seed=$seed
        sh scripts/optimal/adapter.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --seed=$seed
        sh scripts/optimal/prompt.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --seed=$seed
        # Head tuning
        sh scripts/optimal/head.sh 64 0.125 hetero 0.5 8 $dataset --sample_num=20000\ --seed=$seed
        sh scripts/optimal/head.sh 64 1 hetero 0.5 8 $dataset --sample_num=20000\ --seed=$seed
    done
done
