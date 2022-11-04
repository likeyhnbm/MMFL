# client_num client_sample partition_method alpha thread_num dataset custom
for dataset in dataset/cifar100/ 
    do
        for ssl in checkpoints/moco-vit-b-300ep.pth.tar
        do
            # full fine-tuning
            # sh scripts/optimal/pretrain.sh 64 0.125 hetero 0.1 4 $dataset --ssl=$ssl
            # FedPEFT
            sh scripts/optimal/bias.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
            sh scripts/optimal/adapter.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
            sh scripts/optimal/prompt.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
            # Head tuning
            sh scripts/optimal/head.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
        done
    done