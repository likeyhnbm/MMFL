from concurrent.futures import thread
import os

partition_method='homo'
prompt_max_thread = 16
pretrain_max_thread = 8


for client_num in [64,16]:
    for client_sample in [0.125,1]:
        thread_num = client_num * client_sample

        prompt_thread_num = min(prompt_max_thread,thread_num)
        pretrain_thread_num = min(pretrain_max_thread,thread_num)


        prompt_cmd = 'python main.py --partition_method %s --method %s --client_sample %s --client_number %d --thread_number %d --comm_round 50 --lr %s --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_base_patch16_224_in21k --optimizer sgd --sam_mode none --vpt_projection -1' % (partition_method, 'prompt_official', client_sample, client_num, prompt_thread_num, 1e-2)

        pretrain_cmd = 'python main.py --partition_method %s --method %s --client_sample %s --client_number %d --thread_number %d --comm_round 50 --lr %s --epochs 10 --data_dir dataset/cifar100/ --batch_size 64 --vpt_type Deep --vit_type vit_base_patch16_224_in21k --optimizer sgd --sam_mode none --vpt_projection -1' % (partition_method, 'pretrain', client_sample, client_num, pretrain_thread_num, 1e-3)

        os.system(prompt_cmd)
        os.system(pretrain_cmd)