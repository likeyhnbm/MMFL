import torch
import wandb
import logging
from torch.multiprocessing import current_process
from torch.nn.utils import clip_grad_norm_
# from torch.utils.tensorboard import SummaryWriter
# import writer
import numpy as np
from copy import deepcopy


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
        # self.num_classes = client_dict['num_classes']
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None
        self.modality = '-'

        
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
            if self.round >= self.args.warmup_rounds and self.round < (self.args.warmup_rounds + self.args.freeze_rounds):
                self.model.freeze_attn()
            else:
                self.model.unfreeze_attn()

            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            if self.client_index < self.args.vision_client_number:
                # vision client
                self.modality = 'v'
                for g in self.optimizer.param_groups:
                    g['lr'] = self.args.v_lr
                num_samples = len(self.train_dataloader)*self.args.vision_batch_size
            else:
                # vision client
                self.modality = 'l'
                for g in self.optimizer.param_groups:
                    g['lr'] = self.args.l_lr
                num_samples = len(self.train_dataloader)*self.args.language_batch_size

            if self.args.dp:
                self.noise_multiplier = get_noise_multiplier(self.args.epsilon, self.args.delta, self.args.max_grad_norm)
                # logging.info('noise_multiplier:{}'.format(self.noise_multiplier))

            weights, loss = self.train() #if not self.args.dp else self.dp_train()
            acc = self.test()
            # self.model.to('cpu')
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            client_results.append({'weights':deepcopy(weights), 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index, 'modality':self.modality, 'loss': loss})
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
            for batch_idx, batch in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                if self.args.debug and cnt>5:
                    break
                if self.modality == 'l':
                    texts, labels, masks = batch
                    texts, labels, masks = texts.to(self.device), labels.to(self.device), masks.to(self.device)
                    self.optimizer.zero_grad()
                    log_probs = self.model(texts, modality='l', masks=masks)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()

                elif self.modality == 'v':
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    log_probs = self.model(images, modality='v')
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                
                self.optimizer.step()

                batch_loss.append(loss.item())
                cnt+=1
                # logging.info('(client {} cnt {}'.format(self.client_index,cnt))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # self.writer.add_scalar('Loss/client_{}/train'.format(self.client_index), sum(batch_loss) / len(batch_loss), epoch)
                try:
                    logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
                except:
                    logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss), 0, self.client_map[self.round]))
        weights = self.model.cpu().state_dict(modality='vl' if self.args.momentum else self.modality)
        # images, labels = images.to('cpu'), labels.to('cpu')
        loss = sum(epoch_loss) / len(epoch_loss)
        return weights, loss
    
    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            cnt = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.args.debug and cnt>1:
                    break
                if self.modality == 'l':
                    texts, labels, masks = batch
                    texts, labels, masks = texts.to(self.device), labels.to(self.device), masks.to(self.device)

                    pred = self.model(texts, modality='l', masks=masks)
                elif self.modality == 'v':
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    pred = self.model(images, modality='v')
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(labels).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += labels.size(0)
                cnt+=1
            acc = (test_correct / test_sample_number)*100
            logging.info("************* {} Client {} Acc = {:.2f} **************".format('Vision' if self.modality=='v' else 'Language',self.client_index, acc))
        return acc


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
    
class Base_Server():
    def __init__(self,server_dict, args):
        self.v_train_data = server_dict['v_train_data']
        self.v_test_data = server_dict['v_test_data']

        self.l_train_data = server_dict['l_train_data']
        self.l_test_data = server_dict['l_test_data']

        self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        # self.device = 'cpu'
        if 'model_type' in server_dict:
            self.model_type = server_dict['model_type']
        elif 'model' in server_dict:
            self.model = server_dict['model']
        # self.v_num_classes = server_dict['v_num_classes']
        # self.l_num_classes = server_dict['l_num_classes']
        self.v_client_number = server_dict['v_client_number']
        self.l_client_number = server_dict['l_client_number']

        self.v_acc = 0.0
        self.l_acc = 0.0
        self.round = 0
        self.args = args
        self.save_path = server_dict['save_path']

    def run(self, received_info):
        server_outputs = self.operations(received_info)

        # self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        # self.device = 'cuda:0'

        v_acc, l_acc = 0.0, 0.0
        if self.v_client_number > 0:
            self.test_data = self.v_test_data
            v_acc = self.test('v')
        if self.l_client_number > 0:
            self.test_data = self.l_test_data
            l_acc = self.test('l')
            
        acc = v_acc, l_acc
        self.log_info(received_info, acc)
        self.round += 1
        if v_acc > self.v_acc:
            if self.args.save_model:
                torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.v_acc = v_acc
        if l_acc > self.l_acc:
            if self.args.save_model:
                torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.l_acc = l_acc
        return server_outputs
    
    def start(self):
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write('{}\n'.format(self.args))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def log_info(self, client_info, acc):

        v_client_info = [info for info in client_info if info['modality'] == 'v']
        l_client_info = [info for info in client_info if info['modality'] == 'l']

        v_client_acc = sum([c['acc'] for c in v_client_info])/len(v_client_info) if len(v_client_info)!=0 else 0.0
        l_client_acc = sum([c['acc'] for c in l_client_info])/len(l_client_info) if len(l_client_info)!=0 else 0.0
        if not self.args.debug:
            wandb.log({"Test/VisionAccTop1": acc[0], "Test/LanguageAccTop1": acc[1], "Client_Train/LanguageAccTop1": l_client_acc, "Client_Train/VisionAccTop1": v_client_acc, "round": self.round})
        # out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        # with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
        #     out_file.write(out_str)

    def get_losses_balanced_scale(self, client_info):
        losses = {'v':[], 'l': []}
        for c in client_info:
            losses[c['modality']].append(c['loss'])

        v_scale = (1 / (2 * np.var(losses['v'])))
        l_scale = (1 / (2 * np.var(losses['l'])))
        
        # loss_scale = (sum(losses['l']) / len(losses['l'])) / (sum(losses['v']) / len(losses['v']))
        loss_scale = v_scale / l_scale

        return loss_scale 
    
    def get_coeff(self, client_info):

        v_num = sum([x['num_samples'] for x in client_info if x['modality'] == 'v'])
        l_num = sum([x['num_samples'] for x in client_info if x['modality'] == 'l'])

        total_num = 2 * l_num if self.args.balanced and v_num != 0 and l_num!=0 else sum([x['num_samples'] for x in client_info])

        num_scale = l_num / v_num if self.args.balanced and v_num != 0 and l_num!=0 else 1
        loss_scale = self.get_losses_balanced_scale(client_info) if self.args.loss_balanced else 1

        coeffs = []
        for c in client_info:
            w = c['num_samples'] / total_num * num_scale * loss_scale if c['modality'] == 'v' else c['num_samples'] / total_num
            coeffs.append(w)
        
        coeffs = np.array(coeffs)
        coeffs = coeffs / sum(coeffs)

        return coeffs
        



    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])

        if self.args.momentum:
            client_sd = [c['weights'] for c in client_info]

            cw = self.get_coeff(client_info)
            # cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
            ssd = self.model.cpu().state_dict()

            for key in ssd:
                ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])

        else:
            ssd = self.model.cpu().state_dict()
            for key in ssd:
                clients_with_key = [info for info in client_info if key in info['weights'].keys()]
                if len(clients_with_key) > 0: 
                    cur_coeff = self.get_coeff(clients_with_key)
                    client_sd = [c['weights'] for c in clients_with_key]
                    ssd[key] = sum([sd[key]*cur_coeff[i] for i, sd in enumerate(client_sd)])
        #     value = 0
        #     key_sample = 0
        #     for i, sd in enumerate(client_sd):
        #         if key in sd.keys():
        #             if self.args.balanced:
        #                 if client_info[i]['modality'] == 'v':
        #                     key_sample += client_info[i]['num_samples'] * self.args.l_train_data_num / self.args.v_train_data_num
        #                 else:
        #                     key_sample += client_info[i]['num_samples']
        #             else:
        #                 key_sample += client_info[i]['num_samples']

        #     for i, sd in enumerate(client_sd):
        #         if key in sd.keys():
        #             # value += sd[key] * (client_info[i]['num_samples'] / key_sample)
        #             if self.args.balanced:
        #                 if client_info[i]['modality'] == 'v':
        #                     value += sd[key] * (client_info[i]['num_samples'] / key_sample) * self.args.l_train_data_num / self.args.v_train_data_num
        #                 else:
        #                     value += sd[key] * (client_info[i]['num_samples'] / key_sample)
        #             else:
        #                 value += sd[key] * (client_info[i]['num_samples'] / key_sample)
            
        #     ssd[key] = value
            
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def test(self, modality):
        # self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        # self.device = 'cuda:0'
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            cnt = 0
            for batch_idx, batch in enumerate(self.test_data):
                if self.args.debug and cnt>1:
                    break
                if modality == 'l':
                    texts, labels, masks = batch
                    texts, labels, masks = texts.to(self.device), labels.to(self.device), masks.to(self.device)

                    pred = self.model(texts, modality='l', masks=masks)
                elif modality == 'v':
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    pred = self.model(images, modality='v')
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                predicted = predicted.to(labels.device)
                correct = predicted.eq(labels).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += labels.size(0)
                cnt+=1
            acc = (test_correct / test_sample_number)*100
            logging.info("************* {} Server Acc = {:.2f} **************".format('Vision' if modality=='v' else 'Language', acc))
        
        # self.device = 'cpu'
        # x = x.to('cpu')
        # target = target.to('cpu')

        return acc