v_lr=1e-3
l_lr=5e-4
python main.py --v_lr $v_lr --l_lr $l_lr --goal verify --thread_number 8
# python main.py --lr $lr --vision_client_number 0 --goal verify --thread_number 4
# python main.py --lr $lr --language_client_number 0 --goal verify --thread_number 4