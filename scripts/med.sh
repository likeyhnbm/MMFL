# goal=med
# for lr in 5e-4 5e-3 5e-2
# do
#     python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
#     python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
#     python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
# done

goal=med_warm
for lr in 5e-4
do
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --freeze_modality l
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --freeze_modality v
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality v --freeze_modality l --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --warmup_modality l --freeze_modality v --balanced
    # python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
    # python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
done
