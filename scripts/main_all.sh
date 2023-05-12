goal=cifar_all
for lr in 5e-4
do
    # baselines
    # python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 
    # python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 
    # scheduling
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality v
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality l
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality v --freeze_modality l --freeze_rounds 10
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality l --freeze_modality v --freeze_rounds 10
    # balance
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality v --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality l --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality v --freeze_modality l --balanced --freeze_rounds 10
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8  --warmup_modality l --freeze_modality v --balanced --freeze_rounds 10
    # python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 
    # python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 
done