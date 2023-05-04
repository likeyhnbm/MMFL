for v_lr in 5e-4 5e-3 5e-2 
do
    for l_lr in 5e-4 5e-3 5e-2 
    do
        python main.py --v_lr $v_lr --l_lr $l_lr --goal combine --thread_number 8 --warmup_modality v --balanced
    done
done

# python main.py --lr $lr --vision_client_number 0 --goal verify --thread_number 4
# python main.py --lr $lr --language_client_number 0 --goal verify --thread_number 4