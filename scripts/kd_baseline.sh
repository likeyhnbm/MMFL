goal=kd
for lr in 5e-4
do
    # python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --method creamfl --pub_batch_size 256 --vision_batch_size 256 --language_batch_size 256 --kd_weight 0 --interintra_weight 0.01 --comm_round 5
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --method creamfl --pub_batch_size 256 --vision_batch_size 256 --language_batch_size 256 --kd_weight 0.01 --interintra_weight 0

goal=med_kd
for lr in 5e-4
do
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --method creamfl --vision_data_dir dataset/organamnist --language_data_dir dataset/mtsamples --pub_batch_size 256 --vision_batch_size 256 --language_batch_size 256 --kd_weight 0.01
done
done