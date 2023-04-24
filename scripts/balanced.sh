# v_lr=1e-3
# l_lr=5e-4
goal=balanced
for lr in 1e-4 5e-4 1e-3 5e-3
do
    # python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --patch_size 4 --balanced
    # python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --patch_size 4 --balanced
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --patch_size 4 --balanced
done
