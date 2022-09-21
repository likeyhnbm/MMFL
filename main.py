
import torch
import numpy as np
import pdb
import random
import data_preprocessing.data_loader as dl
import argparse

import timm
from models.vpt import build_promptmodel
from models.adapter import build_adapter_model
from models.bias import build_bias_model
from models.vpt_official import build_promptmodel as build_official

import wandb

from torch.multiprocessing import Pool, Process, set_start_method, Queue, Lock

import logging
import os
from collections import defaultdict
import time


# methods
import methods.scratch as scratch
import methods.pretrain as pretrain
import methods.prompt as prompt
import methods.prompt_official as prompt_official
import methods.adapter as adapter
import methods.bias as bias
import methods.shufflenet as shufflenet
import data_preprocessing.custom_multiprocess as cm
from torchvision.models import shufflenet_v2_x0_5


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--method', type=str, default='scratch', metavar='M',
                        help='baseline method')
    parser.add_argument('--prompt_num', type=int, default=10, metavar='N',
                        help='prompt number for vpt') 
    parser.add_argument('--vit_type', type=str, default='vit_small_patch16_384', choices=['vit_small_patch32_224','vit_small_patch16_224', 'vit_base_patch16_224','vit_base_patch16_224_in21k'] , metavar='N',
                        help='type of vpt')  
    parser.add_argument('--vpt_type', type=str, default='Shallow', choices= ['Shallow', 'Deep'], metavar='N',
                        help='type of vpt')
    parser.add_argument('--vpt_projection', type=int, default=-1, metavar='D',
                    help='projection dimension for VIT')
    parser.add_argument('--vpt_drop', type=float, default=0.1, metavar='D',
                    help='projection dimension for VIT')

    parser.add_argument('--model', type=str, default='vit', choices=['resnet18','resnet56','vit'], metavar='M',
                        help='neural network used in training')

    parser.add_argument('--data_dir', type=str, default='data/cifar100',
                        help='data directory: data/cifar100, data/cifar10, or a personal dataset')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=100,
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
    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])
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

def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample<1.0:
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number==0 and num_clients>0:
            clients_per_thread = int(num_clients/args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)]
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
    elif "CropDisease" in path:
        return 'CropDisease'

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    try:
     set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)





    data_name = datapath2str(args.data_dir)

    save_path = './logs/{}_lr{:.0e}_e{}_c{}_{}_{}'.format(
                        args.method, args.lr, args.epochs, args.client_number, data_name, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    args.save_path = save_path
    if not args.debug:
        wandb.init(config=args,project='FedPrompt_new')
        wandb.run.name = save_path.split(os.path.sep)[-1]

    # get data
    img_size = 224 if '224' in args.vit_type else 384
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
         class_num = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size,img_size, args.sample_num)

    mapping_dict = allocate_clients_to_threads(args)
    # pdb.set_trace()
    #init method and model type
    # NOTE Always use fedavg right now
    if args.method=='scratch':
        Server = scratch.Server
        Client = scratch.Client
        Model = timm.create_model
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num,'type':args.vit_type}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num,'type':args.vit_type} for i in range(args.thread_number)]


    elif args.method=='pretrain':
        Server = pretrain.Server
        Client = pretrain.Client
        Model = timm.create_model
        # (args.vit_type,num_classes= class_num, pretrained= True)
        # server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model': Model, 'num_classes': class_num, 'writer':writer}
        # client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
        #                     'client_map':mapping_dict[i], 'model': Model, 'num_classes': class_num, 'writer':writer} for i in range(args.thread_number)]
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num,'type':args.vit_type}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num,'type':args.vit_type} for i in range(args.thread_number)]

    elif args.method=='bias':
        Server = bias.Server
        Client = bias.Client
        Model = build_bias_model

        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num,'type':args.vit_type}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num,'type':args.vit_type} for i in range(args.thread_number)]

    elif args.method=='prompt':
        Server = prompt.Server
        Client = prompt.Client
        basic_model = timm.create_model(args.vit_type, num_classes= class_num, pretrained= True)
        Model = build_promptmodel
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'basic_model':basic_model}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'basic_model':basic_model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method in ['prompt_official', 'head']:
        if args.method == 'head':
            args.prompt_num = 0
        Server = prompt_official.Server
        Client = prompt_official.Client
        basic_model = timm.create_model(args.vit_type, num_classes= class_num, pretrained= True)
        Model = build_official
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'basic_model':basic_model}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'basic_model':basic_model, 'num_classes': class_num} for i in range(args.thread_number)]

    elif args.method=='adapter':
        Server = adapter.Server
        Client = adapter.Client
        basic_model = timm.create_model(args.vit_type, num_classes= class_num, pretrained= True)
        Model = build_adapter_model
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'basic_model':basic_model}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'basic_model':basic_model, 'num_classes': class_num} for i in range(args.thread_number)]       
    elif args.method=='shufflenet':
        Server = shufflenet.Server
        Client = shufflenet.Client
        Model = shufflenet_v2_x0_5
        server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)] 

   
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
            time.sleep(150 * (args.client_number * args.client_sample / args.thread_number)) #  Allow time for threads to start up
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
