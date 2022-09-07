import torch
import wandb
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
from data_preprocessing.sam import SAM



class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)

        self.model = self.model_type(   basic_model=client_dict['basic_model'], 
                                        num_classes=self.num_classes,
                                        reducation_factor=args.reducation_factor
                                    ).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        params = filter(lambda p: p.requires_grad,self.model.parameters())
        if args.optimizer == 'sgd':

                base_optimizer = torch.optim.SGD
                kwargs = dict(lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

        elif args.optimizer == 'adamw':
            base_optimizer = torch.optim.AdamW
            kwargs = dict(lr=self.args.lr, weight_decay=self.args.wd)

        
        if args.sam_mode == 'none':
            self.optimizer = base_optimizer(params, **kwargs)
        elif args.sam_mode == 'asam':
            self.optimizer = SAM(params, base_optimizer, rho=self.args.rho, adaptive=True, **kwargs)
        elif args.sam_mode == 'sam':
            self.optimizer = SAM(params, base_optimizer, rho=self.args.rho, adaptive=False, **kwargs)
        

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(   basic_model=server_dict['basic_model'], 
                                        num_classes=self.num_classes,
                                        reducation_factor=args.reducation_factor
                                    ).to(self.device)      
        if not self.args.debug:
            wandb.watch(self.model)

if __name__ == '__main__':
    from main import add_args, allocate_clients_to_threads
    from methods import prompt
    from models.vpt_official import build_promptmodel
    import timm
    import argparse
    import data_preprocessing.data_loader as dl
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    img_size = 224 if '224' in args.vit_type else 384
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
    class_num = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size,img_size)

    mapping_dict = allocate_clients_to_threads(args)

    Server = prompt.Server
    Client = prompt.Client
    basic_model = timm.create_model(args.vit_type, num_classes= class_num, pretrained= True)
    Model = build_promptmodel

    server_dict = {'train_data':train_data_global, 'test_data': test_data_global, 'model_type': Model, 'num_classes': class_num, 'basic_model':basic_model}
    client_dict = [{'train_data':train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                            'client_map':mapping_dict[i], 'model_type': Model, 'basic_model':basic_model, 'num_classes': class_num} for i in range(args.thread_number)]
   
   
   
   
    client = Client(client_dict[0], args)
    client.model.obtain_prompt()