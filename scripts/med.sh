# goal=med
# for lr in 5e-4 5e-3 5e-2
# do
#     python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
#     python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
#     python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
# done

goal=med_new_all
for lr in 5e-4
do
    # baselines
    python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
    python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
    # vanilla
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
    # scheduling
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --freeze_modality l --freeze_rounds 10
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --freeze_modality v --freeze_rounds 10
    # balance
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --freeze_modality l --balanced --freeze_rounds 10
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --freeze_modality v --balanced --freeze_rounds 10
    # python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
    # python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
done
