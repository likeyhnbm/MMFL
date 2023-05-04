import torch
import wandb
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
# from data_preprocessing.dpsgd import DPSGD

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)

        self.model = self.model_type(args).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        params = filter(lambda p: p.requires_grad,self.model.parameters())


        if args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.v_lr)
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.v_lr)
    

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(args).to(self.device)
    
