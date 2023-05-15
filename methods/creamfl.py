import torch
import wandb
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
from copy import deepcopy
import operator
import torch.nn as nn
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

        self.public_loader = client_dict['pub_loader']

    def load_client_state_dict(self, server_state_dict):
            
        self.model.load_state_dict(server_state_dict)
    
    def load_global_features(self, global_img_feature, global_txt_feature, distill_index):
            
        self.global_img_feature = global_img_feature.to(self.device)
        self.global_txt_feature = global_txt_feature.to(self.device)
        self.distill_index = distill_index

    def run(self, received_info):
        client_results = []
        self.load_global_features(received_info['global_img_feature'], received_info['global_txt_feature'], received_info['distill_index'])
        # try:
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info['weight'])
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

            weights, loss = self.train() #if not self.args.dp else self.dp_train()
            acc = self.test()
            vectors, _ = self.generate_logits(self.public_loader)


            # self.model.to('cpu')
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            client_results.append({'weights': deepcopy(weights), 'num_samples':num_samples,'acc':acc, 'client_index':self.client_index, 'modality':self.modality, 'loss': loss, 'vecs': deepcopy(vectors)})
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
        # logging.info(''
        self.old_model = deepcopy(self.model)
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
        
        # distill_dict = {int(b): a for a, b in enumerate(self.distill_index)}
        # for idx, (images, captions, txt_masks, index) in (enumerate(self.public_loader)):
        #     self.optimizer.zero_grad()
        #     d_idx = operator.itemgetter(*index.tolist())(distill_dict)  # batchidx
        #     if self.modality == 'v':
        #         images = images.to(self.device)
        #         im_feature = self.model(images, 'v', feat_out=True)
        #         target_feature = self.global_img_feature[d_idx, :].type_as(im_feature)
                
        #         with torch.no_grad():
        #             old_im_feature = self.old_model(images, 'v', feat_out=True)

        #         logits_inter = torch.div(torch.matmul(im_feature, self.global_txt_feature.T), 0.5)
        #     elif self.modality == 'l':
        #         captions = captions.to(self.device)
        #         txt_masks = txt_masks.to(self.device) 
        #         im_feature = self.model(captions, 'l', txt_masks, feat_out=True)
        #         target_feature = self.global_txt_feature[d_idx, :].type_as(im_feature)
        #         # neg
        #         with torch.no_grad():
        #             old_im_feature = self.old_model(captions, 'l', txt_masks, feat_out=True)

        #         logits_inter = torch.div(torch.matmul(im_feature, self.global_img_feature.T), 0.5)

        #     labels_inter = torch.tensor(d_idx).to(self.device)
        #     loss_inter = self.criterion(logits_inter, labels_inter)

        #     # pos
        #     pos = torch.sum(im_feature * target_feature, dim=-1)
        #     pos = pos.reshape(-1, 1)
        #     # neg
        #     # neg = cos(im_feature, old_im_feature)
        #     neg = torch.sum(im_feature * old_im_feature, dim=-1)
        #     logits = torch.cat((pos, neg.reshape(-1, 1)), dim=1)

        #     logits /= 0.5  # temperature
        #     labels = torch.zeros(images.size(0)).to(self.device).long()

        #     loss_moon = self.criterion(logits, labels)

        #     loss = (loss_moon + loss_inter) * self.args.interintra_weight

        #     loss.backward()
        #     nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 2)
        #     self.optimizer.step()

        weights = self.model.cpu().state_dict()
        # images, labels = images.to('cpu'), labels.to('cpu')
        loss_ = sum(epoch_loss) / len(epoch_loss)
        return weights, loss_
    
    def generate_logits(self, dataloader):
        vec, idx = self.extract_pub_feature(dataloader)
        if self.modality == 'v':
            return {'img': vec, 'txt': None}, idx
        elif self.modality == 'l':
            return {'img': None, 'txt': vec}, idx
        else:
            assert False

    def extract_pub_feature(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        feature = []
        distill_index = []
        # iterate batch
        for idx, (images, captions, txt_masks, index) in (enumerate(dataloader)):
            with torch.no_grad():
                if self.modality == 'v':
                    images = images.to(self.device)
                    im_feature = self.model(images, 'v', feat_out=True)

                elif self.modality == 'l':
                    captions = captions.to(self.device)
                    txt_masks = txt_masks.to(self.device) 
                    im_feature = self.model(captions, 'l', txt_masks, feat_out=True)

                im_feature = im_feature.cpu().detach()
                feature.append(im_feature)
                distill_index.extend(index)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and idx == 1:
                #     break

        feature = torch.cat(feature, dim=0)
        # print(f'feature {feature.shape} labels {labels.shape}')

        self.model.cpu()
        return feature, distill_index
    

class Server(Base_Server):
    def __init__(self,server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(args).to(self.device)

        # self.optimizer
        self.public_loader = server_dict['pub_loader']

        if args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.p_lr)
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.p_lr)

    def generate_public_logit(self):
        img_feature, txt_feature = [], []
        distill_index = []
        self.model.eval()
        self.model.to(self.device)
        for idx, (images, captions, txt_masks, index) in enumerate(self.public_loader):
            with torch.no_grad():
                images = images.to(self.device)  # [bs, 3, 224, 224]
                captions = captions.to(self.device)  # [bs, seq_len]
                txt_masks = txt_masks.to(self.device) 

                # output = self.model(images, captions, , capt_lens)
                out_img = self.model(images, 'v', feat_out=True)
                out_txt = self.model(captions, 'l', txt_masks, feat_out=True)

                out_img = out_img.cpu().detach()
                out_txt = out_txt.cpu().detach()

                img_feature.append(out_img)
                txt_feature.append(out_txt)
                distill_index.extend(index)

        self.global_img_feature = torch.concat(img_feature, dim=0)
        self.global_txt_feature = torch.concat(txt_feature, dim=0)
        # print(self.global_txt_feature.shape, self.global_img_feature.shape)
        self.distill_index = distill_index
        del img_feature, txt_feature
        torch.cuda.empty_cache()

    def start(self):
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write('{}\n'.format(self.args))

        self.generate_public_logit()
        info = {
            'weight': self.model.cpu().state_dict(),
            'global_img_feature': self.global_img_feature,
            'global_txt_feature': self.global_txt_feature,
            'distill_index': self.distill_index
        }
        return [info for x in range(self.args.thread_number)]


    def run(self, received_info):
        self.operations(received_info)

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

        self.generate_public_logit()

        info = {
            'weight': self.model.cpu().state_dict(),
            'global_img_feature': self.global_img_feature,
            'global_txt_feature': self.global_txt_feature,
            'distill_index': self.distill_index
        }
        server_output = [info for x in range(self.args.thread_number)]

        return server_output
    
    # def aggregation(self, client_info):


    def distill(self, img_vec, txt_vec, img_num, txt_num, distill_index):

        self.model.train()
        self.model.to(self.device)

        client_loss_cri = nn.MSELoss()

        def aggregation(i_vec=img_vec, t_vec=txt_vec, i_num=img_num, t_num=txt_num):
            
            if len(i_vec) > 0: 
                contrastive_w = []
                for vec in i_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                    logits = torch.matmul(vec, self.global_txt_feature.to(vec.device).T)  # [50000, 50000]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                    contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                for i in range(len(i_vec)):
                    i_vec[i] = (i_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                i_vec = torch.sum(torch.cat(i_vec, dim=0), dim=0)  # aggregated image vectors
            else:
                i_vec = []

            if len(t_vec) > 0: 
                contrastive_w = []
                for vec in t_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                    logits = torch.matmul(vec, self.global_img_feature.to(vec.device).T)  # [50000, 50000]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                    contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                for i in range(len(t_vec)):
                    t_vec[i] = (t_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                t_vec = torch.sum(torch.cat(t_vec, dim=0), dim=0)  # aggregated text vectors

            else:
                t_vec= None
            
            self.global_img_feature.cpu().detach()
            self.global_txt_feature.cpu().detach()

            return i_vec, t_vec

        # aggregation
        img_vec, txt_vec = aggregation()

        self.img_vec = img_vec
        self.txt_vec = txt_vec

        distill_dict = {int(b): a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
        # distill
        for idx, (images, captions, txt_mask, index) in enumerate(self.public_loader):
            images = images.to(self.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.device)  # [bs, seq_len]
            # caption_lens = caption_lens.to(self.device)

            # output = self.model(images, captions, txt_mask, index)
            loss = 0

            def code_sim(output, target):
                output = output.sum(axis=1) if len(output.shape) == 3 else output
                target = target.type_as(output)

                return client_loss_cri(output, target.type_as(output))

            if self.img_vec is not None and len(self.img_vec) > 0:
                out_img = self.model(images, 'v', feat_out=True)
                d_idx = operator.itemgetter(*index.tolist())(distill_dict)  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                loss += self.args.kd_weight * code_sim(out_img, target_img)
            if self.txt_vec is not None and len(self.txt_vec) > 0:
                out_txt = out_img = self.model(captions, 'l', feat_out=True)
                d_idx = operator.itemgetter(*index.tolist())(distill_dict)  # idx of the current batch
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.args.kd_weight * code_sim(out_txt, target_txt)


            self.optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 2)

            self.optimizer.step()

    
    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = self.get_coeff(client_info)

        # cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
        ssd = self.model.cpu().state_dict()

        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        
        self.model.load_state_dict(ssd)

        img_vec, img_num = [], []
        txt_vec, txt_num = [], []

        for info in client_info:
            _vec = info['vecs']

            if _vec['img'] is not None:
                img_vec.append(_vec['img'].to(self.device))
                img_num.append(info['num_samples'])
                # print(f'img_vec {_vec["img"].shape}')
            if _vec['txt'] is not None:
                txt_vec.append(_vec['txt'].to(self.device))
                txt_num.append(info['num_samples'])
                # print(f'txt_vec {_vec["txt"].shape}')

        self.distill(img_vec, txt_vec, img_num, txt_num, self.distill_index)

        del img_vec
        del txt_vec
        # client_sd = [c['weights'] for c in client_info]
        
        # cw = self.get_coeff(client_info)
        # # cw = [c['num_samples']/sum([x['num_samples'] for x in client_info]) for c in client_info]
        # ssd = self.model.state_dict()
        # for key in ssd:
        #     ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])

        # self.model.load_state_dict(ssd)
        # if self.args.save_client:
        #     for client in client_info:
        #         torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        # return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]    
    
