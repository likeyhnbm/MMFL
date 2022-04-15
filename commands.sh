#!/bin/bash 

# main
python main.py --method fedavg --client_number 16 --comm_round 2 --lr 0.01 --epochs 20
python main.py --method gradaug --client_number 16 --comm_round 2 --lr 0.01 --epochs 20 --width 0.8 --mult 1.75 --num_subnets 2
python main.py --method moon --client_number 16 --comm_round 2 --lr 0.01 --epochs 20 --mu 1.0
python main.py --method feddepth --client_number 16 --comm_round 2 --lr 0.01 --epochs 20 --stoch_depth 0.9
python main.py --method mixup --client_number 16 --comm_round 2 --lr 0.01 --epochs 20 --beta 0.1
python main.py --method ours --client_number 16 --comm_round 2 --lr 0.01 --epochs 20 --width 0.25 --mu 0.45

# python main.py --method fedavg --client_number 64 --thread_number 16 --client_sample 0.25 --comm_round 100 --lr 0.01 --epochs 20
# ptyhon main.py --method fedavg --client_number 64 --thread_number 16 --client_sample 0.25 --comm_round 100 --lr 0.01 --epochs 20 --data_dir data/cifar10
