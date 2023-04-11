for lr in 5e-2 1e-3 5e-4 1e-4
do
    python main.py --v_lr $lr --language_client_number 0 --goal lr --thread_number 4 --optimizer sgd
    python main.py --l_lr $lr --vision_client_number 0 --goal lr --thread_number 4 --optimizer sgd
    python main.py --v_lr $lr --l_lr $lr --vision_client_number 0 --goal lr --thread_number 4 --optimizer sgd
done
