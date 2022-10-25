import torch
import numpy as np
from pyhessian.hessian import hessian
# import models.resnet as fedavg
# import models.resnet_gradaug as resnet_fl
# import models.resnet_feddepth as resnet_stochdepth
# import models.resnet_ours as resnet_lip
# import torch.nn.functional as F
# import models.ComputePostBN as ComputePostBN
import pyhessian.density_plot as dp
import logging
import argparse
import matplotlib.pyplot as plt
import copy
from statistics import mean
import time
from pathlib import Path
import pickle
from matplotlib import cm
import data_preprocessing.data_loader as dl
import yaml
import random

import torch
from models.vpt_official import VPT_ViT, build_promptmodel
from models.adapter import build_adapter_model
from models.bias import build_bias_model
import timm

def get_model(type, weights):
    vit_type = 'vit_base_patch16_224_in21k'
    class_num = 100
    if type in ['adapter', 'prompt', 'bias', 'head']:
        basic_model = timm.create_model(vit_type, num_classes= class_num, pretrained= True)
        if type =='adapter':
            reducation_factor = 8
            model = build_adapter_model(basic_model=basic_model, 
                                        num_classes=class_num,
                                        reducation_factor=reducation_factor
                                    )
        elif type in ['prompt', 'head']:
            prompt_num = 10 if type=='prompt' else 0
            model = build_promptmodel(basic_model=basic_model, 
                                num_classes=class_num,
                                vpt_type='Deep',
                                prompt_num=prompt_num,
                                edge_size=224,
                                patch_size=16,
                                projection = -1,
                                prompt_drop_rate= 0.1,
                            )
        elif type=='bias':
            model = build_bias_model(vit_type, num_classes=class_num)
    if type in ['pretrain']:
        model = timm.create_model(vit_type, num_classes= class_num, pretrained= True)
    model.load_state_dict(weights, strict=False)

    return model

def getEigen(args, logger):
    # logger.info('{}\t{}\t{}\n'.format(args.yaml['method']['value'], args.yaml['data_dir']['value'], args.model_dir))
    device = 'cuda:0'
        # model = fedavg.resnet18(200)
    weights = torch.load(args.model_dir,  map_location = torch.device(device))
    model = get_model(args.method, weights)
    # model.load_state_dict(torch.load(args.model_dir,  map_location = torch.device(device)))
    model =  torch.nn.DataParallel(model).cuda()
    # model.cuda()

    # create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # get dataset 
    _,_, train_data_global, test_data_global, _, _, _,_ = dl.load_partition_data('dataset/cifar100', 'homo',0.5, 16, 64)
    train_loader = train_data_global
    client_num = 'GS'
    # model = ComputePostBN.ComputeBN(model, train_loader, 32 if 'cifar' in args.yaml['data_dir']['value'] else 224, 'cuda:0')
    model.eval()

    if args.single_batch:
        # for illustrate, we only use one batch to do the tutorial
        for inputs, targets in train_loader:
            break

        # we use cuda to make the computation fast
        inputs, targets = inputs.cuda(), targets.cuda()

        hessian_comp = hessian(model, criterion, data=(inputs, targets), device=device)
    else:
        hessian_comp = hessian(model, criterion, dataloader=train_loader, device=device)

    # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
    # print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])

    # Now let's compute the top 2 eigenavlues and eigenvectors of the Hessian
    # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    # print("The top two eigenvalues of this model are: %.4f %.4f"% (top_eigenvalues[-1],top_eigenvalues[-2]))
    logger.info('{}'.format(args.model_dir))
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    logger.info('***Top Eigenvalues: {}'.format(top_eigenvalues))
    # logger.info('***Top Eigenvalues: {}'.format([0]))
    
    trace, diag = hessian_comp.trace()
    logger.info('***Trace: {}'.format(np.mean(trace)))
    logger.info('***Diag: {}'.format(diag))
    np.save('{}/diag'.format(args.save_dir), diag)

    if args.test:
        test(model, test_data_global, client_num)
    # weight_mag = torch.mean(torch.stack([torch.norm(x[1].detach()) for x in model.named_parameters()])).item()
    # logger.info('***Weight Magnitude: {}'.format(weight_mag))
    # density_eigen, density_weight = hessian_comp.density()
    # print('Plotting Density...')
    # dp.get_esd_plot(density_eigen, density_weight, '{}/{}'.format(args.save_dir, 'density.pdf'))
    print('Plotting Loss Space...')
    # plotLoss(model, top_eigenvector, criterion, inputs, targets, args)
    plotLoss(model, top_eigenvector, criterion, train_loader, args)
    print('Done')

# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def get_Z(model_orig,  model_perb, directions, alphas):
    idx = 0
    for m_orig, m_perb in zip(model_orig.parameters(), model_perb.parameters()):
        if m_orig.requires_grad:
            m_perb.data = m_orig.data + alphas[0] * directions[0][idx] + alphas[1] * directions[1][idx]
            idx += 1
    return model_perb

def plotLoss(model, top_eigenvector, criterion, train_loader, args):
    lams = np.linspace(-1.0, 1.0, 11).astype(np.float32)
    # if len(top_eigenvector)==1:
    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors
    # 2D
    model_perb = copy.deepcopy(model)
    # model_perb =  torch.nn.DataParallel(model_perb).cuda()
    # model_perb.eval()

    # loss_list = []
    # for lam in lams:
    #     model_perb = get_params(model, model_perb, top_eigenvector[0], lam)
    #     avg_list = []
    #     for inputs, targets in train_loader:
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #         avg_list.append(criterion(model_perb(inputs), targets).item())
    #         if args.single_batch:
    #             break
    #     loss_list.append(mean(avg_list))
    # with open('{}/loss_list.p'.format(args.save_dir), 'wb') as fp:   #Pickling
    #     pickle.dump(loss_list, fp)
    # plt.figure()
    # plt.plot(lams, loss_list)
    # plt.ylabel('Loss')
    # plt.title('Loss landscape perturbed based on top Hessian eigenvector')
    # plt.savefig('{}/{}'.format(args.save_dir, '2D.pdf'))
    # else:
    # 3D
    model_perb = copy.deepcopy(model)
    # model_perb =  torch.nn.DataParallel(model_perb).cuda()
    # model_perb.eval()
    x, y = np.meshgrid(lams, lams, sparse=True)
    z = np.zeros((len(lams), len(lams)))
    for xidx in range(len(lams)):
        for yidx in range(len(lams)):
            lam = [x[0,xidx], y[yidx,0]]
            model_perb = get_Z(model, model_perb, top_eigenvector, lam)
            avg_list = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                avg_list.append(criterion(model_perb(inputs), targets).item())
                if args.single_batch:
                    break
            z[xidx, yidx] = mean(avg_list)
    np.save('{}/z'.format(args.save_dir), z)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Perturbation X')
    ax.set_ylabel('Perturbation Y')
    ax.set_zlabel('Loss')
    for ii in range(0,360,30):
        ax.view_init(elev=10., azim=ii)
        plt.savefig('{}/movie{}.png'.format(args.save_dir, ii))

def precomputed_plot(args):
    # with open('{}/loss_list.p'.format(args.model_dir), 'rb') as ll:
    #     loss_list = pickle.load(ll)
    lams = np.linspace(-1.0, 1.0, 11).astype(np.float32)
    # # lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    # # lams = lams[10:-10]
    # # loss_list = loss_list[10:-10]
    # plt.figure()
    # plt.plot(lams, loss_list)
    # plt.ylabel('Loss')
    # plt.title('Loss landscape perturbed based on top Hessian eigenvector')
    # plt.savefig('{}/{}'.format(args.model_dir, '2D.pdf'))

    x, y = np.meshgrid(lams, lams, sparse=True)
    plt.figure()
    z = np.load('{}/z.npy'.format(args.model_dir))
    # z = z[10:-10, 10:-10]
    ax = plt.axes(projection='3d')
    MAX_Z = 2.5
    ax.set_zlim(0, MAX_Z)
    my_col = cm.cividis(z/MAX_Z)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
    mappable.set_array(np.array([0,MAX_Z]))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='cividis', edgecolor='none', facecolors=my_col)
    ax.set_xlabel(r'$\epsilon_{0}$')
    ax.set_ylabel(r'$\epsilon_{1}$')
    ax.set_zlabel('Loss')
    plt.colorbar(mappable,fraction=0.02, pad=0.02, orientation="vertical")
    for ii in range(0,360,30):
        ax.view_init(elev=10., azim=ii)
        plt.savefig('{}/movie{}.pdf'.format(args.model_dir, ii))

def test(model, test_dataloader, client_index):
    model.cuda()
    model.eval()

    test_correct = 0.0
    test_loss = 0.0
    test_sample_number = 0.0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dataloader):
            x = x.cuda()
            target = target.cuda()

            pred = model(x)
            # loss = criterion(pred, target)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(target).sum()

            test_correct += correct.item()
            # test_loss += loss.item() * target.size(0)
            test_sample_number += target.size(0)
        acc = (test_correct / test_sample_number)*100
        logging.info("************* Client {} Acc = {:.2f} **************".format(client_index, acc))
    return acc

def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hess(model_dir):
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.model_dir = model_dir
    args.single_batch = False
    args.precomputed = False
    args.test = True
    set_random_seed(1)

    if args.precomputed:
        precomputed_plot(args)
    else:
        args.save_dir = 'pyhessian/logs/Vf_{}_{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.model_dir.split('/')[1])
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='{}/test.log'.format(args.save_dir), filemode='a', level=logging.INFO)
        logger = logging.getLogger('Hessian')
        with open('{}/{}'.format('/'.join(args.model_dir.split('/')[:-1]), 'config.yaml'), 'rb') as f:
            args.yaml = yaml.safe_load(f.read())
        getEigen(args, logger)

if __name__ == "__main__":
    # python compute_hessian.py --model_dir wandb/offline-run-20210730_134115-suboqsu3/files/client_15.pt --single_batch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='wandb/run-20210730_053421-1nn9arbu/files/server.pt', metavar='N',
                        help='federated learning method')
    parser.add_argument('--single_batch', action='store_true', default=False,
                        help='test pretrained model')
    parser.add_argument('--precomputed', action='store_true', default=False,
                        help='test pretrained model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Get global test accuracy of model')
    args, unknown = parser.parse_known_args()
    set_random_seed(1)

    if args.precomputed:
        precomputed_plot(args)
    else:
        method = args.model_dir.split('/')[-1].replace('.pt','')
        args.method = method
        args.save_dir = 'pyhessian/logs/{}_{}'.format(time.strftime("%Y%m%d-%H%M%S"), method)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='{}/test.log'.format(args.save_dir), filemode='a', level=logging.INFO)
        logger = logging.getLogger('Hessian')
        # with open('{}/{}'.format('/'.join(args.model_dir.split('/')[:-1]), 'config.yaml'), 'rb') as f:
            # args.yaml = yaml.safe_load(f.read())
        getEigen(args, logger)
        # getEigen('logs/cifar100-wideresnet/checkpoint_single.pt')
        # logs/cifar100-wideresnet/checkpoint_mutual.pt