goal=kd
for lr in 5e-4
do
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --method creamfl
done

goal=med_kd
for lr in 5e-4
do
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --method creamfl --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples
done