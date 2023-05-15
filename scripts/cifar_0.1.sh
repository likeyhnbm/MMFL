goal=cifar_0.1

for lr in 5e-4
    do
        # baselines
        python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --seed $seed
        python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --seed $seed
        # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --seed $seed
        python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --method creamfl --pub_batch_size 256 --vision_batch_size 256 --language_batch_size 256 --kd_weight 0.01 --interintra_weight 0 --seed $seed
        # scheduling
        # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality v
        python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality l --seed $seed
        # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality v --freeze_modality l --freeze_rounds 10
        # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality l --freeze_modality v --freeze_rounds 10
    done