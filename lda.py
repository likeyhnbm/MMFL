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
from models.vlmo import VLMo

def get_models(type, weights):

    
    if type == 'uni':
        v_model = VLMo(32, patch_size=4)
        l_model = VLMo(32, patch_size=4)
        v_model.load_state_dict(weights[0])
        l_model.load_state_dict(weights[1])
    else:
        v_model = l_model = VLMo(32, patch_size=4)
        v_model.load_state_dict(weights)
        

    return v_model, l_model


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
def get_avg_feat(model_dirs, v_test_data, l_test_data, type):
    # if args.mutual:
    #     model = resnet56_mutual(100, KD=True)
    # else:
    #     model = resnet56(100, KD=True)
    device = 'cuda:0'
    # model.load_state_dict(torch.load(model_dir,  map_location = torch.device(device)))


    weights = torch.load(model_dirs[0], map_location=torch.device(device)) if type=='ours' else [torch.load(model_dirs[0], map_location=torch.device(device)), torch.load(model_dirs[1], map_location=torch.device(device))]
    v_model, l_model = get_models(type, weights)


    # v
    v_model.eval()
    v_model.to(device)
    # model = torch.nn.DataParallel(model)

    # test_correct = 0.0
    # test_loss = 0.0
    # test_sample_number = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(v_test_data):
            x = x.to(device)
            target = target.to(device)

            prior = v_model(x, 'v', None, True)

            preds.append(prior.detach())
            labels.append(['v'] * len(x))

        # avg_pred = torch.mean(torch.stack(preds), dim=1).cpu().numpy()


    l_model.eval()
    l_model.to(device)

    with torch.no_grad():
        for batch_idx, (x, target, mask) in enumerate(l_test_data):
            x = x.to(device)
            target = target.to(device)
            mask=mask.to(device)

            prior = v_model(x, 'l', mask, True)

            preds.append(prior.detach())
            labels.append(['l'] * len(x))


    type='ours'

    model_dirs = ['/home/guangyu/projects/MMFL/logs/ours_lr5e-04_5e-04_e10_c16_16_Cifar100_AGNews_2023-05-15_14-41-19/server.pt']
    weights = torch.load(model_dirs[0], map_location=torch.device(device)) if type=='ours' else [torch.load(model_dirs[0], map_location=torch.device(device)), torch.load(model_dirs[1], map_location=torch.device(device))]
    v_model, l_model = get_models(type, weights)


    # v
    v_model.eval()
    v_model.to(device)
    # model = torch.nn.DataParallel(model)

    # test_correct = 0.0
    # test_loss = 0.0
    # test_sample_number = 0.0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(v_test_data):
            x = x.to(device)
            target = target.to(device)

            prior = v_model(x, 'v', None, True)

            preds.append(prior.detach())
            labels.append(['v_o'] * len(x))

        # avg_pred = torch.mean(torch.stack(preds), dim=1).cpu().numpy()


    l_model.eval()
    l_model.to(device)

    with torch.no_grad():
        for batch_idx, (x, target, mask) in enumerate(l_test_data):
            x = x.to(device)
            target = target.to(device)
            mask=mask.to(device)

            prior = v_model(x, 'l', mask, True)

            preds.append(prior.detach())
            labels.append(['l_o'] * len(x))



    feats = torch.cat(preds, dim=0).cpu().numpy()

    return feats, labels

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
    parser.add_argument('--vision_data_dir', type=str, default='dataset/cifar100',
                        help='data directory')
    parser.add_argument('--language_data_dir', type=str, default='dataset/agnews',
                        help='data directory')
    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
    parser.add_argument('--client_number', type=int, default=1, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--type', type=str, default='uni', metavar='N',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()

    model_dirs = ['/home/guangyu/projects/MMFL/logs/ours_lr5e-04_1e-03_e10_c16_0_Cifar100_AGNews_2023-05-15_11-55-39/server.pt', '/home/guangyu/projects/MMFL/logs/ours_lr1e-03_5e-04_e10_c0_16_Cifar100_AGNews_2023-05-15_12-45-35/server.pt'] if args.type == 'uni' else ['/home/guangyu/projects/MMFL/logs/ours_lr5e-04_5e-04_e10_c16_16_Cifar100_AGNews_2023-05-15_14-41-19/server.pt']
    avg_feats = {} 
    # with open('{}/wandb-metadata.json'.format(model_dir)) as f:
    #     json_args = json.load(f)
    # if 'mutual' in json_args['args'][1]:
    #     args.mutual=True
    # else:
    #     args.mutual=False


    v_train_data_num, v_test_data_num, v_train_data_global, v_test_data_global, v_data_local_num_dict, v_train_data_local_dict, v_test_data_local_dict,\
        v_class_num = dl.load_partition_data(args.vision_data_dir, args.partition_method, args.partition_alpha, 1, args.batch_size,32, -1)



    l_train_data_num, l_test_data_num, l_train_data_global, l_test_data_global, l_data_local_num_dict, l_train_data_local_dict, l_test_data_local_dict,\
            l_class_num = dl.load_partition_data(args.language_data_dir, args.partition_method, args.partition_alpha, 1, args.batch_size,32, -1)

    idx = 0
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.manifold import TSNE
    # from sklearn.
    import pandas as pd
    import seaborn as sns

    type2label = {
        'v': 0,
        'l': 1,
        'v_o': 2,
        'l_o': 3,
    }
    label2type = {v:k for k,v in type2label.items()}

    formal_name = {
        'v': 'Uni-FedAVG: Image Features',
        'l': 'Uni-FedAVG: Text Features',
        'v_o': 'FedUniT: Image Features',
        'l_o': 'FedUniT: Text Features',
    }

    xs = []
    ys = []
    classes = []

    type = args.type
    print(type)
    avg_feat, labels = get_avg_feat( model_dirs, v_test_data_global, l_test_data_global, type)
    labels = np.array(labels).flatten()
    labels = [j for i in labels for j in i]
    # idx = int(infile.split('.')[0].split('_')[1])
    # avg_feats[idx] = avg_feat
    idx += 1
    for sample, cls in zip(avg_feat, labels):
        xs.append(sample)
        ys.append(type2label[cls])
        # classes.append(cls)

            # tsne = TSNE(n_components=2, verbose=1)
            # tsne_results = tsne.fit_transform(avg_feat)
    xs = np.array(xs)
    ys = np.array(ys)
    lda = LinearDiscriminantAnalysis(n_components=2)
    # tsne = TSNE(n_components=2, verbose=1)
    results = lda.fit_transform(xs,ys)

    df=pd.DataFrame([formal_name[label2type[y]] for y in ys], columns=['Features'])
    df['tsne-2d-one'] = results[:,0]
    df['tsne-2d-two'] = results[:,1]
    # df['class'] = classes
    df.to_csv('method.csv')

    df = pd.read_csv('method.csv')

    # df = df[df['Features'].isin(['Uni-FedAVG: Image Features', 'Uni-FedAVG: Text Features'])]

    plt.figure(figsize=(5,4))
    colors = sns.color_palette("hls", 4)
    color_dict = {formal_name[label2type[i]]: colors[i] for i in range(4)}

    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Features",
        # hue_order = ['normal', 'abnormal'],
        palette=color_dict,
        data=df,
        legend='brief',
        alpha=0.8
    )
    g.set(xticks=[], yticks=[], xlabel=None, ylabel=None)
    # g.legend( ['normal','abnormal'])
    # g.legend_.set_title(None)
    # g.legend_.set_label(['normal', 'abnormal'])

    plt.savefig(os.path.join('',"lda_avg.png"), dpi=300, bbox_inches='tight', pad_inches=0.05)



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