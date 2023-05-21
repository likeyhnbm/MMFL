lr=5e-4
goal=get_weights
seed=1
# 16, 0.5 CIFAR
# 16, 0.5 MED
# 16, 0.1 CIFAR
# baselines
# python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --seed $seed --save_model
# python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --seed $seed --save_model
# scheduling
python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality l --seed $seed --save_model