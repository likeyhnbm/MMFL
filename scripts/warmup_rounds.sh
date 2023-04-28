v_lr=5e-4
l_lr=5e-4
for r in 5 15 20
do
    python main.py --v_lr $v_lr --l_lr $l_lr --goal warmup_rounds --thread_number 8 --warmup_modality v --patch_size 4 --warmup_rounds $r
    python main.py --v_lr $v_lr --l_lr $l_lr --goal warmup_rounds --thread_number 8 --warmup_modality l --patch_size 4 --warmup_rounds $r
done