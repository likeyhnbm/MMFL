import torch
import wandb
import logging
from methods.base import Base_Client, Base_Server

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)

        self.model = self.model_type(basic_model=client_dict['basic_model'], num_classes=self.num_classes,VPT_type=args.vpt_type,Prompt_Token_num=args.prompt_num).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        
class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(basic_model=server_dict['basic_model'], num_classes=self.num_classes,VPT_type=args.vpt_type,Prompt_Token_num=args.prompt_num).to(self.device)