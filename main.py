
import torch
import numpy as np
import pdb
import random
import data_preprocessing.data_loader as dl
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import timm
# from models.vpt import build_promptmodel
# from models.adapter import build_adapter_model
# from models.bias import build_bias_model
# from models.vpt_official import build_promptmodel as build_official
from models.vlmo import build_vlmo, build_separate, build_shared

import wandb

from torch.multiprocessing import Pool, Process, set_start_method, Queue, Lock

import logging
import os
from collections import defaultdict
import time




# methods
# import methods.scratch as scratch
import methods.ours as ours
# import methods.pretrain as pretrain
# import methods.prompt as prompt
# import methods.prompt_official as prompt_official
# import methods.adapter as adapter
# import methods.bias as bias
# import methods.shufflenet as shufflenet
import methods.creamfl as creamfl
import data_preprocessing.custom_multiprocess as cm
from data_preprocessing.coco import get_pub_loader_kwargs
from torch.utils.data import DataLoader
# from torchvision.models import shufflenet_v2_x0_5


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--method', type=str, default='ours', metavar='M',
                        help='baseline method')
    parser.add_argument('--goal', type=str, default='run', metavar='M',
                        help='goal of this run')
    parser.add_argument('--prompt_num', type=int, default=10, metavar='N',
                        help='prompt number for vpt') 
    parser.add_argument('--vit_type', type=str, default='vit_base_patch16_224_in21k' , metavar='N',
                        help='type of vpt')  
    parser.add_argument('--vpt_type', type=str, default='Shallow', choices= ['Shallow', 'Deep'], metavar='N',
                        help='type of vpt')
    parser.add_argument('--ssl', type=str, default='none', metavar='N',
                        help='type of self-supervised backbone')
    parser.add_argument('--vpt_projection', type=int, default=-1, metavar='D',
                    help='projection dimension for VIT')
    parser.add_argument('--vpt_drop', type=float, default=0.1, metavar='D',
                    help='projection dimension for VIT')

    parser.add_argument('--model', type=str, default='vit', choices=['resnet18','resnet56','vit'], metavar='M',
                        help='neural network used in training')

    parser.add_argument('--vision_data_dir', type=str, default='dataset/cifar100',
                        help='data directory: dataset/cifar100 or a personal dataset')
    parser.add_argument('--language_data_dir', type=str, default='dataset/agnews',
                        help='data directory: dataset/agnews, or a personal dataset')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--vision_client_number', type=int, default=16, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--language_client_number', type=int, default=16, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--vision_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--language_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--v_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--l_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=70,
                        help='how many round of communications we shoud use')

    parser.add_argument('--pretrained_backbone', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default=0.001, metavar='MU',
                        help='mu (default: 0.001)')
    parser.add_argument('--rho', type=float, default=0.05, metavar='PHO',
                        help='pho for SAM (default: 0.05)')

    parser.add_argument('--width', type=float, default=0.7, metavar='WI',
                        help='minimum width for mutual training')
    parser.add_argument('--mult', type=float, default=1.0, metavar='MT',
                        help='multiplier for mutual training')
    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets for mutual training')
    parser.add_argument('--localbn', action='store_true', default=False,
                        help='Keep local BNs each round')
    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')
    parser.add_argument('--thread_number', type=int, default=16, metavar='NN',
                        help='number of threads in a distributed cluster')
    parser.add_argument('--client_sample', type=float, default=0.25, metavar='MT',
                        help='Fraction of clients to sample')
    parser.add_argument('--resolution_type', type=int, default=0,
                        help='Specifies the resolution list used')
    parser.add_argument('--stoch_depth', default=0.5, type=float,
                    help='stochastic depth probability')
    parser.add_argument('--beta', default=0.0, type=float,
                    help='hyperparameter beta for mixup')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='reduce the sampler number for debugging')    
    parser.add_argument('--optimizer', default='adamw',choices= ['sgd','adamw'],type=str,
                    help='selection of optimizer')            
    parser.add_argument('--stat', action='store_true', default=False,
                    help='show the state of model')  
    parser.add_argument('--dp', action='store_true', default=False,
                help='Apply differential privacy')  
    parser.add_argument('--delta', default=1e-3, type=float,
                    help='hyperparameter delta for DP')
    parser.add_argument('--epsilon', default=5, type=float,
                    help='hyperparameter epsilon for DP')
    parser.add_argument('--max_grad_norm', default=1, type=float,
                    help='hyperparameter grad_norm for DP')
    parser.add_argument('--freeze_all', action='store_true', default=False,
                    help='freeze the entire model') 
    parser.add_argument('--save_model', action='store_true', default=False,
                help='save the server model') 

    parser.add_argument('--sam_mode', type=str, default='none', choices= ['asam', 'sam', 'none'], metavar='N',
                        help='type of sam')

    parser.add_argument('--sample_num', type=int, default=-1, metavar='N',
                        help='how many sample will be trained in total. -1 for no reduce')

    parser.add_argument('--reducation_factor', type=int, default=8, metavar='N',
                    help='reducation_factor for adapter')
    
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed')

    parser.add_argument('--vocab_size', type=int, default=30522,
                        help='Vocabulary size (default: 30522)')
    parser.add_argument('--max_text_len', type=int, default=40,
                        help='Maximum text length (default: 40)')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Image size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size (default: 4)')
    parser.add_argument('--warmup_modality', type=str, default='vl',
                        help='warm up modality')
    parser.add_argument('--warmup_rounds', type=int, default=10,
                        help='warm up rounds')
    parser.add_argument('--freeze_modality', type=str, default='vl',
                        help='warm up modality')
    parser.add_argument('--freeze_rounds', type=int, default=0,
                        help='warm up rounds')
    parser.add_argument('--balanced', action='store_true', default=False,
                help='balanced between modalities') 
    parser.add_argument('--loss_balanced', action='store_true', default=False,
                help='balanced between modalities') 
    parser.add_argument('--momentum', action='store_false', default=True,
                help='balanced between modalities') 
    
    # CreamFL
    parser.add_argument('--pub_data_dir', type=str, default='dataset/coco/images/val2017/',
                        help='public data directory: dataset/cifar100 or a personal dataset')
    parser.add_argument('--pub_anno_dir', type=str, default='dataset/coco/annos/annotations/captions_val2017.json',
                        help='public annotation directory: dataset/cifar100 or a personal dataset')
    parser.add_argument('--num_pub_samples', type=int, default=5000,
                        help='public samples number')
    parser.add_argument('--pub_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max_length', type=int, default=40, metavar='N',
                        help='max_length of txt data')
    parser.add_argument('--p_lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--interintra_weight', type=float, default=0.5,
                        help='inter intra loss weight')
    parser.add_argument('--kd_weight', type=float, default=0.3, help='coefficient of kd')

    args = parser.parse_args()

    return args

# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    
    global client
    ci = q.get()
    set_random_seed(ci[1].seed)
    client = Client(ci[0], ci[1])
    logging.info("Initialized.")
    # client.server = 

def run_clients(received_info):
    # try:
        # received_info, save_path = info
        # client, received_info = client_args
        # glo.set_value('writer', SummaryWriter(log_dir=save_path))
        return client.run(received_info)
    # except KeyboardInterrupt:
    #     logging.info('exiting')
    #     return None

# def get_round_client_list(args):
#     round_client_lists = []
#     for i in range(args.warmup_rounds):
#         if args.warmup_modality == 'vl':
#             num_clients = int(args.client_number * args.client_sample)
#             client_list = random.sample(range(args.client_number), num_clients)
#         elif args.warmup_modality == 'v':
#             num_clients = int(args.client_number * args.client_sample)
#             client_list = random.sample(range(args.vision_client_number), num_clients)
def get_round_modality(args):
    round_modalities = []
    for i in range(args.warmup_rounds):
        round_modalities.append(args.warmup_modality)
    for i in range(args.freeze_rounds):
        round_modalities.append(args.freeze_modality)
    for i in range(args.warmup_rounds + args.freeze_rounds, args.comm_round):
        round_modalities.append('vl')

    return round_modalities

def is_modality(args, id, modality):
    if modality == 'v':
        return  id < args.vision_client_number
    elif modality == 'l':
        return  id > args.vision_client_number
    elif modality == 'vl':
        return True

def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    round_modalities = get_round_modality(args)
    for round in range(args.comm_round):
        if args.client_sample<1.0:
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        # client_list = [idx for idx in client_list if is_modality(args, idx, round_modalities[round])]
        # num_clients = len(client_list)
        if num_clients % args.thread_number==0 and num_clients>0:
            clients_per_thread = int(num_clients/args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread) if is_modality(args, client_list[x], round_modalities[round]) ]
                mapping_dict[c].append(idxs)
        else:
            logging.info("############ WARNING: Sampled client number not divisible by number of threads ##############")
            break
    return mapping_dict

# MODEL_DICT = {
#     'resnet56': resnet56,
#     'resnet18': resnet18,
#     'vit': timm.create_model,
#     # 'prompt': 
# }

# DATA_DICT = {
#     'dataset/cifar-100-python/' : 'Cifar100',
#     "" : 'CropDisease'
# }

def datapath2str(path):
    if "cifar-100" in path or "cifar100" in path:
        return 'Cifar100'
    elif "agnews" in path:
        return 'AGNews'
    elif "CropDisease" in path:
        return 'CropDisease'
    else:
        return os.path.split(path)[-1]

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":

    
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    
    try:
     set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed(args.seed)

    data_name = datapath2str(args.vision_data_dir) + '_' + datapath2str(args.language_data_dir)

    save_path = './logs/{}_lr{:.0e}_{:.0e}_e{}_c{}_{}_{}_{}'.format(
                        args.method, args.v_lr,args.l_lr, args.epochs, args.vision_client_number, args.language_client_number, data_name, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    args.save_path = save_path
    
    args.client_number = args.vision_client_number + args.language_client_number

    if not args.debug:
        wandb.init(config=args,project='mmfl', entity='ucf_vision')
        wandb.run.name = save_path.split(os.path.sep)[-1]

    # get data
    # img_size = 224 if '224' in args.vit_type else 384

    if args.vision_client_number > 0:
        v_train_data_num, v_test_data_num, v_train_data_global, v_test_data_global, v_data_local_num_dict, v_train_data_local_dict, v_test_data_local_dict,\
            v_class_num = dl.load_partition_data(args.vision_data_dir, args.partition_method, args.partition_alpha, args.vision_client_number, args.vision_batch_size,args.img_size, args.sample_num)

    else:
        v_train_data_num, v_test_data_num, v_train_data_global, v_test_data_global, v_data_local_num_dict, v_train_data_local_dict, v_test_data_local_dict,\
            v_class_num = None, None, None, None, None, None, None, None
    if args.language_client_number > 0:
        l_train_data_num, l_test_data_num, l_train_data_global, l_test_data_global, l_data_local_num_dict, l_train_data_local_dict, l_test_data_local_dict,\
            l_class_num = dl.load_partition_data(args.language_data_dir, args.partition_method, args.partition_alpha, args.language_client_number, args.language_batch_size,args.img_size, args.sample_num)
    else:
        l_train_data_num, l_test_data_num, l_train_data_global, l_test_data_global, l_data_local_num_dict, l_train_data_local_dict, l_test_data_local_dict,\
            l_class_num = None, None, None, None, None, None, None, None
        
    if args.vision_client_number == args.client_number:
        train_data_local_dict = v_train_data_local_dict
        test_data_local_dict = v_test_data_local_dict
    elif args.language_client_number == args.client_number:
        train_data_local_dict = l_train_data_local_dict
        test_data_local_dict = l_test_data_local_dict
    else:
        l_train_data_local_dict = {k + args.vision_client_number: v for k,v in l_train_data_local_dict.items()}
        train_data_local_dict = [v_train_data_local_dict, l_train_data_local_dict]
        train_data_local_dict = {key: loader for dicts in train_data_local_dict for key, loader in dicts.items()}

        l_test_data_local_dict = {k + args.vision_client_number: v for k,v in l_test_data_local_dict.items()}
        test_data_local_dict = [v_test_data_local_dict, l_test_data_local_dict]
        test_data_local_dict = {key: loader for dicts in test_data_local_dict for key, loader in dicts.items()}

    args.v_class_num = v_class_num
    args.l_class_num = l_class_num

    args.v_train_data_num = v_train_data_num
    args.l_train_data_num = l_train_data_num


    mapping_dict = allocate_clients_to_threads(args)
    # pdb.set_trace()
    #init method and model type
    # NOTE Always use fedavg right now
    if args.method=='ours':
        Server = ours.Server
        Client = ours.Client
        Model = build_vlmo
        server_dict = { 'v_train_data':v_train_data_global,
                        'l_train_data':l_train_data_global,
                        'v_test_data': v_test_data_global, 
                        'l_test_data': l_test_data_global, 
                        'model_type': Model, 
                        'v_num_classes': v_class_num,
                        'l_num_classes': l_class_num, 
                        'v_client_number': args.vision_client_number,
                        'l_client_number': args.language_client_number,
                        }
            
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model} for i in range(args.thread_number)]
        
    elif args.method=='shared':
        Server = ours.Server
        Client = ours.Client
        Model = build_shared
        server_dict = { 'v_train_data':v_train_data_global,
                        'l_train_data':l_train_data_global,
                        'v_test_data': v_test_data_global, 
                        'l_test_data': l_test_data_global, 
                        'model_type': Model, 
                        'v_num_classes': v_class_num,
                        'l_num_classes': l_class_num, 
                        'v_client_number': args.vision_client_number,
                        'l_client_number': args.language_client_number,
                        }
            
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model} for i in range(args.thread_number)]

    elif args.method=='creamfl':
        Server = creamfl.Server
        Client = creamfl.Client
        Model = build_separate
        pub_loader_kwargs = get_pub_loader_kwargs(args.pub_data_dir, args.pub_anno_dir, args.num_pub_samples, args.img_size, args.max_length, args.pub_batch_size)
        pub_loader = DataLoader(**pub_loader_kwargs)
        server_dict = { 'v_train_data':v_train_data_global,
                        'l_train_data':l_train_data_global,
                        'v_test_data': v_test_data_global, 
                        'l_test_data': l_test_data_global, 
                        'model_type': Model, 
                        'v_num_classes': v_class_num,
                        'l_num_classes': l_class_num, 
                        'v_client_number': args.vision_client_number,
                        'l_client_number': args.language_client_number,
                        'pub_loader': pub_loader
                        }
            
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'pub_loader': pub_loader} for i in range(args.thread_number)]

    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')



    if args.pretrained:
        print('Pretrained')
        server_dict['save_path'] = "best.pt"
        server = Server(server_dict, args)
        server.model.load_state_dict(torch.load('logs/2022-03-21 20:56:10.626330_fedavg_e20_c16/server.pt'))
        acc = server.test()
    else:
        # init server
        server_dict['save_path'] = save_path
        if not os.path.exists(server_dict['save_path']):
            os.makedirs(server_dict['save_path'])
        server = Server(server_dict, args)
        server_outputs = server.start()

        param_num = get_parameter_number(server.model)
        if not args.debug:
            wandb.log({"Params/Total": param_num["Total"], "Params/Trainable":param_num["Trainable"]})
        # Start Federated Training
        #init nodes
        client_info = Queue(32)
        # clients = {}
        for i in range(args.thread_number):
            client_info.put((client_dict[i], args))
            # clients[i] = Client(client_dict[i], args)

        # Start server and get initial outputs
        pool = cm.MyPool(args.thread_number, init_process, (client_info, Client))

        if args.debug:
            time.sleep(10 * (args.client_number * args.client_sample / args.thread_number))
        else:
            time.sleep(120 * (args.client_number * args.client_sample / args.thread_number)) #  Allow time for threads to start up
        for r in range(args.comm_round):
            logging.info('************** Round: {} ***************'.format(r))
            round_start = time.time()
            client_outputs = pool.map(run_clients, server_outputs)
            client_outputs = [c for sublist in client_outputs for c in sublist]
            server_outputs = server.run(client_outputs)
            round_end = time.time()
            if not args.debug:
                wandb.log({"Round Time": round_end-round_start, "round": r})
            out_str = ' Round {} Time: {}s \n'.format(r, round_end-round_start)
            logging.info(out_str)
            with open('{}/out.log'.format(args.save_path), 'a+') as out_file:
                out_file.write(out_str)
            # time.sleep(10)
        pool.close()
        pool.join()
