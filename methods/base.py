import torch
import wandb
import logging
from torch.multiprocessing import current_process
from torch.nn.utils import clip_grad_norm_
# from torch.utils.tensorboard import SummaryWriter
# import writer
import numpy as np


def get_noise_multiplier(epsilon, delta, max_grad_norm):
    """
        grads: [N, d]
    """


    # # sensitivity
    s = 2 * max_grad_norm

    sigma = s * np.sqrt(2 * np.log(1.25/delta) / epsilon)

    return sigma

class Base_Client():
    def __init__(self, client_dict, args):
        self.train_data = client_dict['train_data']
        self.test_data = client_dict['test_data']
        self.device = 'cuda:{}'.format(client_dict['device'])
        if 'model_type' in client_dict:
            self.model_type = client_dict['model_type']
        elif 'model' in client_dict:
            self.model = client_dict['model']
        # self.writer = SummaryWriter(args.save_path)
        self.num_classes = client_dict['num_classes']
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None

        
    def set_server(self,server):
        self.server = server
    
    def load_client_state_dict(self, server_state_dict):
        if self.args.localbn:
            server_dict = {k: v for k, v in server_state_dict.items() if 'bn' not in k}
            self.model.load_state_dict(server_dict, strict=False)
        else:
            self.model.load_state_dict(server_state_dict)
    
    def run(self, received_info):
        client_results = []
        # try:
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader)*self.args.batch_size

            if self.args.dp:
                self.noise_multiplier = get_noise_multiplier(self.args.epsilon, self.args.delta, self.args.max_grad_norm)
                # logging.info('noise_multiplier:{}'.format(self.noise_multiplier))

            weights = self.train() #if not self.args.dp else self.dp_train()
            acc = self.test()
            client_results.append({'weights':weights, 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()
        # except:
        #     print(self.client_index, self.round)

        self.round += 1
        return client_results
        
    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # _writer = glo.get_value("writer")
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            cnt = 0 
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                if self.args.debug and cnt>5:
                    break
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                if self.args.dp:
                    for param in self.model.parameters():
                        if param.requires_grad:
                            clip_grad_norm_(param.grad, max_norm=self.args.max_grad_norm)
                self.optimizer.step()
                if self.args.dp:
                    for param in self.model.parameters():
                        if param.requires_grad:
                            with torch.no_grad():
                                param = param + self.args.lr * torch.normal(mean=0, std=self.noise_multiplier, size=param.size()).to(self.device)

                batch_loss.append(loss.item())
                cnt+=1
                # logging.info('(client {} cnt {}'.format(self.client_index,cnt))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        return weights

    # def dp_train(self):
    #     # train the local model
    #     self.model.to(self.device)
    #     self.model.train()
    #     # _writer = glo.get_value("writer")
    #     epoch_loss = []
    #     for epoch in range(self.args.epochs):
    #         with BatchMemoryManager(
    #                                     data_loader=self.train_dataloader, 
    #                                     max_physical_batch_size=self.args.batch_size, 
    #                                     optimizer=self.optimizer
    #                                 ) as memory_safe_data_loader:
    #             batch_loss = []
    #             cnt = 0 
    #             for batch_idx, (images, labels) in enumerate(self.train_dataloader):
    #                 # logging.info(images.shape)
    #                 if self.args.debug and cnt>5:
    #                     break
    #                 images, labels = images.to(self.device), labels.to(self.device)
    #                 self.optimizer.zero_grad()
    #                 log_probs = self.model(images)
    #                 loss = self.criterion(log_probs, labels)
    #                 loss.backward()
    #                 self.optimizer.step()
    #                 batch_loss.append(loss.item())
    #                 cnt+=1
    #                 # logging.info('(client {} cnt {}'.format(self.client_index,cnt))
    #             if len(batch_loss) > 0:
    #                 epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #                 # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
    #                 logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
    #                                                                             epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
    #                 if self.args.dp:
    #                     delta = self.args.delta
    #                     epsilon, best_alpha = self.privacy_engine.accountant.get_privacy_spent(delta=delta)
    #                     logging.info(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")

    #     weights = self.model.cpu().state_dict()
    #     return weights
    def dp_train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # _writer = glo.get_value("writer")
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            cnt = 0 
            for batch_idx, batch in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                if self.args.debug and cnt>5:
                    break
                for param in self.model.parameters():
                    param.accumulated_grads = []

                images, labels = batch    

                for image, label in zip(images, labels):
                    image, label = image.to(self.device).unsqueeze(0), label.to(self.device).unsqueeze(0)
                    
                    self.optimizer.zero_grad()
                    log_probs = self.model(image)
                    loss = self.criterion(log_probs, label)
                    batch_loss.append(loss.item())
                    loss.backward()

                    for param in self.model.parameters():
                        per_sample_grad = param.grad.detach().clone()
                        clip_grad_norm_(per_sample_grad, max_norm=self.args.max_grad_norm)  # in-place
                        param.accumulated_grads.append(per_sample_grad)  
                for param in self.model.parameters():
                    param.grad = torch.stack(param.accumulated_grads, dim=0)

                for param in self.model.parameters():
                    param = param - self.args.lr * param.grad
                    param += torch.normal(mean=0, std=self.noise_multiplier * self.args.max_grad_norm)
                    
                    param.grad = 0  # Reset for next iteration

                
                cnt+=1
                    # logging.info('(client {} cnt {}'.format(self.client_index,cnt))
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
                    logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

        weights = self.model.cpu().state_dict()
        return weights


    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            cnt = 0
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                if self.args.debug and cnt>1:
                    break
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                cnt+=1
            acc = (test_correct / test_sample_number)*100
            logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
        return acc
    
class Base_Server():
    def __init__(self,server_dict, args):
        self.train_data = server_dict['train_data']
        self.test_data = server_dict['test_data']
        # self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        self.device = 'cpu'
        if 'model_type' in server_dict:
            self.model_type = server_dict['model_type']
        elif 'model' in server_dict:
            self.model = server_dict['model']
        self.num_classes = server_dict['num_classes']
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.save_path = server_dict['save_path']

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        try:
            self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
            acc = self.test()
            self.device = 'cpu'
        except:
            self.device = 'cpu'
            acc = self.test()
            
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            if self.args.save_model:
                torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        return server_outputs
    
    def start(self):
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write('{}\n'.format(self.args))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def log_info(self, client_info, acc):
        client_acc = sum([c['acc'] for c in client_info])/len(client_info)
        if not self.args.debug:
            wandb.log({"Test/AccTop1": acc, "Client_Train/AccTop1": client_acc, "round": self.round})
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def test(self):
        # self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        # self.device = 'cuda:0'
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            cnt = 0
            for batch_idx, (x, target) in enumerate(self.test_data):
                if self.args.debug and cnt>1:
                    break
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                predicted = predicted.to(target.device)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                cnt+=1
            acc = (test_correct / test_sample_number)*100
            logging.info("************* Server Acc = {:.2f} **************".format(acc))
        
        # self.device = 'cpu'
        return acc