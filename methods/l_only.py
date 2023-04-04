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
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # _writer = glo.get_value("writer")
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            cnt = 0 
            for batch_idx, (texts, labels, masks) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                if self.args.debug and cnt>5:
                    break
                texts, labels, masks = texts.to(self.device), labels.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(texts, modality='l', masks=masks)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # if self.args.dp:
                #     for param in self.model.parameters():
                #         if param.requires_grad:
                #             clip_grad_norm_(param.grad, max_norm=self.args.max_grad_norm)
                #     # state_dict = {}
                #     for k, param in self.model.named_parameters():
                #         if param.requires_grad:
                #             with torch.no_grad():
                #                 param.grad += torch.normal(mean=0, std=self.noise_multiplier, size=param.size()).to(self.device)

                #     # self.model.load_state_dict(state_dict, strict=False)
                
                self.optimizer.step()


                batch_loss.append(loss.item())
                cnt+=1
                # logging.info('(client {} cnt {}'.format(self.client_index,cnt))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        # images, labels = images.to('cpu'), labels.to('cpu')
        return weights
    

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(args).to(self.device)