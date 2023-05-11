goal=med_0.1
for lr in 5e-4
do
    # baselines
    python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --partition_alpha 0.1
    python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --partition_alpha 0.1
    # vanilla
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --partition_alpha 0.1
    # scheduling
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --partition_alpha 0.1
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --partition_alpha 0.1
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --freeze_modality l --freeze_rounds 10 --partition_alpha 0.1
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --freeze_modality v --freeze_rounds 10 --partition_alpha 0.1
    # balance
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --balanced --partition_alpha 0.1
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --balanced --partition_alpha 0.1
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --freeze_modality l --balanced --freeze_rounds 10 --partition_alpha 0.1
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --freeze_modality v --balanced --freeze_rounds 10 --partition_alpha 0.1
    # python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
    # python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
done
