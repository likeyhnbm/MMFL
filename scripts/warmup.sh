v_lr=1e-3
l_lr=5e-4
python main.py --v_lr $v_lr --l_lr $l_lr --goal warmup --thread_number 8 --warmup_modality v
python main.py --v_lr $v_lr --l_lr $l_lr --goal warmup --thread_number 8 --warmup_modality l
