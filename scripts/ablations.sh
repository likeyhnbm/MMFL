# client_num client_sample partition_method alpha thread_num dataset custom
for dataset in dataset/cifar100/ 
    do
        for ssl in checkpoints/mae_pretrain_vit_base.pth checkpoints/moco-vit-b-300ep.pth.tar
        do
            # full fine-tuning
            sh scripts/search/pretrain.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
            # FedPEFT
            sh scripts/search/bias.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
            sh scripts/search/adapter.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
            sh scripts/search/prompt.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
            # # Head tuning
            sh scripts/search/head.sh 64 0.125 hetero 0.1 8 $dataset --ssl=$ssl
        done
    done