v_lr=1e-3
l_lr=5e-4
goal=optimizer
python main.py --v_lr $v_lr --l_lr $l_lr --goal $goal --thread_number 8 --optimizer sgd
python main.py --l_lr $l_lr --vision_client_number 0 --goal $goal --thread_number 4 --optimizer sgd
python main.py --v_lr $v_lr --language_client_number 0 --goal $goal --thread_number 4 --optimizer sgd