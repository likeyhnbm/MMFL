goal=pathmnist
for lr in 5e-2 1e-3 5e-4 1e-4 5e-3 1e-2
do
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --vision_data_dir dataset/pathmnist
    python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --vision_data_dir dataset/pathmnist
    # python main.py --l_lr $lr --vision_client_number 0 --goal lr --thread_number 4
    
done
