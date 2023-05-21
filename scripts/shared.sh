goal=all_shared


lr=5e-4

for seed in 1 716 506
do
# 16, 0.5 CIFAR
python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared --seed $seed

# 16, 0.5 MED
# baselines
# scheduling
python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared  --seed $seed --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples

# 16, 0.1 CIFAR
# scheduling
python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared --seed $seed --partition_alpha 0.1


# 16, 0.1 MED
# baselines
python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared  --seed $seed --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --partition_alpha 0.1

# 32, 0.5 CIFAR
# baselines
python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared --seed $seed --client_sample 0.125  --vision_client_number 32 --language_client_number 32

# 32, 0.5 MED
# baselines
# scheduling
python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared --seed $seed --client_sample 0.125  --vision_client_number 32 --language_client_number 32 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
done

















# for seed in 716 506 
# do
#     for lr in 5e-4
#     do
#         # baselines
#         # python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --seed $seed
#         # python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --seed $seed
#         # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --seed $seed
#         python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --method creamfl --pub_batch_size 256 --vision_batch_size 256 --language_batch_size 256 --kd_weight 0.01 --interintra_weight 0 --seed $seed
#         # scheduling
#         # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared
#         python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared --seed $seed
#         # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared --freeze_modality l --freeze_rounds 10
#         # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --method shared --freeze_modality v --freeze_rounds 10
#     done
# done