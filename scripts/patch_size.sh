v_lr=1e-3
l_lr=5e-4
goal=patch
for lr in 1e-2 5e-3 1e-3 5e-4 1e-4
do
    python main.py --v_lr $lr --l_lr $lr --goal $goal --thread_number 8 --patch_size 2 --vision_batch_size 128 --language_batch_size 128
    python main.py --v_lr $lr --language_client_number 0 --goal $goal --thread_number 4 --patch_size 2 --vision_batch_size 256 --language_batch_size 256
    python main.py --l_lr $lr --vision_client_number 0 --goal $goal --thread_number 4 --patch_size 2 --vision_batch_size 256 --language_batch_size 256
done
