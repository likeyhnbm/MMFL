# https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
import numpy as np
import argparse
import os
from models.resnet import resnet56, resnet18
# from models.resnet_mutual_fl import resnet56 as resnet56_mutual
# from models.resnet_mutual_fl import resnet18 as resnet18_mutual
import torch
import data_preprocessing.data_loader as dl
import json
import matplotlib.pyplot as plt
import timm
from models.adapter import build_adapter_model
from models.bias import build_bias_model
from models.vpt_official import build_promptmodel

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
        elif type in ['prompt']:
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
    if type in ['pretrain', 'head']:
        model = timm.create_model(vit_type, num_classes= class_num, pretrained= True)
        if type == 'head':
            new = {}
            for k,v in weights.items():
                if 'basic_model' in k:
                    new_k = k.replace('basic_model.',"")
                    new.update({new_k:v})
            weights = new
    model.load_state_dict(weights, strict=False)

    return model


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
    x: A num_examples x num_features matrix of features.

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
    A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
    The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)

def my_forward(model, x, type):

    if type in ['adapter','prompt']:
        x = model.forward_features(x)
        x0 = model.basic_model.forward_head(x, pre_logits=True)
        x = model.basic_model.forward_head(x, pre_logits=False)
    elif type in ['bias']:
        x = model.model.forward_features(x)
        x0 = model.model.forward_head(x, pre_logits=True)
        x = model.model.forward_head(x, pre_logits=False)
    elif type in ['head','pretrain']:
        x = model.forward_features(x)
        x0 = model.forward_head(x, pre_logits=True)
        x = model.forward_head(x, pre_logits=False)

    return x0, x


###################################################
def get_avg_feat(model_dir, test_data, type):
    # if args.mutual:
    #     model = resnet56_mutual(100, KD=True)
    # else:
    #     model = resnet56(100, KD=True)
    device = 'cuda:0'
    # model.load_state_dict(torch.load(model_dir,  map_location = torch.device(device)))

    
    weights = torch.load(model_dir)
    model = get_model(type, weights)
    model.eval()
    model.to(device)
    # model = torch.nn.DataParallel(model)

    test_correct = 0.0
    test_loss = 0.0
    test_sample_number = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        ###
        # if args.mutual:
        #     model.apply(lambda m: setattr(m, 'width_mult', 1.0))
        # ###
        for batch_idx, (x, target) in enumerate(test_data):
            x = x.to(device)
            target = target.to(device)

            prior, pred = my_forward(model,x, type)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(target).sum()

            test_correct += correct.item()
            # test_loss += loss.item() * target.size(0)
            test_sample_number += target.size(0)
            preds.append(prior.detach())
            labels.append(target.cpu().numpy())
        acc = (test_correct / test_sample_number)*100
        # avg_pred = torch.mean(torch.stack(preds), dim=1).cpu().numpy()
        avg_pred = torch.cat(preds, dim=0).cpu().numpy()
        print("************* {} Acc = {:.2f} **************".format(model_dir.split('/')[-1], acc))
    return avg_pred, labels

if __name__ == "__main__":
    SAVE_NAME='cka2'
    seed = 1
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default='wandb/offline-run-20210711_182725-1tn29d0x', metavar='N',
                        help='federated learning method')
    parser.add_argument('--data_dir', type=str, default='dataset/cifar100',
                        help='data directory')
    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
    parser.add_argument('--client_number', type=int, default=1, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()

    model_dir = '.'
    avg_feats = {}
    # with open('{}/wandb-metadata.json'.format(model_dir)) as f:
    #     json_args = json.load(f)
    # if 'mutual' in json_args['args'][1]:
    #     args.mutual=True
    # else:
    #     args.mutual=False

    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
         class_num = dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size)

    idx = 0
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns
    for infile in os.listdir(model_dir):
        # if infile.endswith('.pt') and 'server' not in infile:
        if infile.endswith('pretrain.pt'):
            type = infile.split('.')[0]
            print(type)
            avg_feat, labels = get_avg_feat('{}/{}'.format(model_dir, infile), test_data_global, type)
            labels = np.array(labels).flatten()
            labels = [j for i in labels for j in i]
            # idx = int(infile.split('.')[0].split('_')[1])
            # avg_feats[idx] = avg_feat
            idx += 1

            tsne = TSNE(n_components=2, verbose=1)
            tsne_results = tsne.fit_transform(avg_feat)

            df=pd.DataFrame(labels, columns=['y'])
            df['tsne-2d-one'] = tsne_results[:,0]
            df['tsne-2d-two'] = tsne_results[:,1]

            plt.figure(figsize=(16,10))
            colors = sns.color_palette("hls", 100)
            color_dict = {i: colors[i] for i in range(100)}

            g = sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue="y",
                # hue_order = ['normal', 'abnormal'],
                palette=color_dict,
                data=df,
                legend=False,
                alpha=0.8
            )
            g.set(xticks=[], yticks=[], xlabel=None, ylabel=None)
            # g.legend( ['normal','abnormal'])
            # g.legend_.set_title(None)
            # g.legend_.set_label(['normal', 'abnormal'])

            plt.savefig(os.path.join('t-sne',type+".png"))



    # cka_matrix = np.full( (len(avg_feats), len(avg_feats)), -1.0)
    # for key, feat in avg_feats.items():
    #     print(key)
    #     for key2, feat2 in avg_feats.items():
    #         # f = np.mean(feat, 0)
    #         # f = torch.from_numpy(feat)
    #         # f2 = torch.from_numpy(feat2)
    #         # f2 = np.mean(feat2, 0)
    #         # cka_matrix[key][key2] = np.dot(f, f2)/(np.linalg.norm(f)*np.linalg.norm(f2))
    #         # cka_matrix[key][key2] = torch.mean(torch.nn.functional.cosine_similarity(f, f2)).item()
    #         cka_matrix[key][key2] = cka(gram_linear(feat), gram_linear(feat2))
    # print('***************** MEAN: {}'.format(np.mean(cka_matrix)))
    # np.save('{}/{}'.format(model_dir, SAVE_NAME), cka_matrix)
    # plt.figure()
    # plt.matshow(cka_matrix, vmin=0.0, vmax=1.0)
    # plt.colorbar()
    # plt.savefig('{}/{}.png'.format(model_dir, SAVE_NAME))
    print('DONE')