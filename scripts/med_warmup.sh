v_lr=5e-4
l_lr=5e-4
python main.py --v_lr $v_lr --l_lr $l_lr --goal warmup --thread_number 8 --warmup_modality v --patch_size 4 --vision_data_dir dataset/organamnist
python main.py --v_lr $v_lr --l_lr $l_lr --goal warmup --thread_number 8 --warmup_modality l --patch_size 4 --vision_data_dir dataset/organamnist


