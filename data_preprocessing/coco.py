import os

import torch
from torchvision import datasets

try:
    import ujson as json
except ImportError:
    import json

from PIL import Image
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from glob import glob

import numpy as np

from torchvision import transforms
from functools import partial
from torch.utils import data
import operator


class CocoCaptionsCap(Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        ids (list, optional): list of target caption ids
        extra_annFile (string, optional): Path to extra json annotation file (for training)
        extra_ids (list, optional): list of extra target caption ids (for training)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        instance_annFile (str, optional): Path to instance annotation json (for PMRP computation)
    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root='dir where images are',
                                    annFile='json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """
    def __init__(self, root, annFile, ids=None,
                 extra_annFile=None, extra_ids=None,
                 transform=None, target_transform=None,
                 instance_annFile=None, client=-1):
        self.root = os.path.expanduser(root)
        if extra_annFile:
            self.coco = COCO()
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                  list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]
        self.transform = transform
        self.target_transform = target_transform

        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])

        iid_to_cls = {}
        if instance_annFile:
            for ins_file in glob(instance_annFile + '/instances_*'):
                with open(ins_file) as fin:
                    instance_ann = json.load(fin)
                for ann in instance_ann['annotations']:
                    image_id = int(ann['image_id'])
                    code = iid_to_cls.get(image_id, [0] * 90)
                    code[int(ann['category_id']) - 1] = 1
                    iid_to_cls[image_id] = code

                seen_classes = {}
                new_iid_to_cls = {}
                idx = 0
                for k, v in iid_to_cls.items():
                    v = ''.join([str(s) for s in v])
                    if v in seen_classes:
                        new_iid_to_cls[k] = seen_classes[v]
                    else:
                        new_iid_to_cls[k] = idx
                        seen_classes[v] = idx
                        idx += 1
                iid_to_cls = new_iid_to_cls

                if self.all_image_ids - set(iid_to_cls.keys()):
                    # print(f'Found mismatched! {self.all_image_ids - set(iid_to_cls.keys())}')
                    print(f'Found mismatched! {len(self.all_image_ids - set(iid_to_cls.keys()))}')

        self.iid_to_cls = iid_to_cls
        self.n_images = len(self.all_image_ids)
    
    def reduce_samples(self, num_samples=1000):

        sampled = np.random.choice(len(self), num_samples, replace=False)

        self.ids = list(operator.itemgetter(*sampled)(self.ids))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']
        caption = annotation['caption']  # language caption

        path = coco.loadImgs(image_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(caption)
        else:
            target = caption

        return img, target['input_ids'], target['attention_mask'], index

    def __len__(self):
        return len(self.ids)
    
def txt_transform(max_length=40):
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case="uncased" in 'bert_base_uncased'
    )
    
    train_transform = partial(tokenizer, padding='max_length', max_length=max_length, truncation=True)

    return train_transform

def img_transform(img_size=32):

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),
        # transforms.RandomCrop(img_size, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_transform
    

def get_pub_loader_kwargs(root, anno_path, num_pub_samples=1000, img_size=32, max_length=40, batch_size=512):

    v_transform = img_transform(img_size=img_size)
    l_transform = txt_transform(max_length=max_length)

    dataset = CocoCaptionsCap(root, anno_path, transform=v_transform, target_transform=l_transform)
    if num_pub_samples > -1:
        dataset.reduce_samples(num_pub_samples)

    # dl = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, persistent_workers=True, collate_fn=public_collate_fn)

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': False,
        'drop_last': False,
        'num_workers': 0,
        # 'persistent_workers': True,
        'collate_fn': public_collate_fn
    }

    return loader_kwargs

def public_collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
      data: list of (image, sentence) tuple.
        - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
        - sentence: torch tensor of shape (?); variable length.

    Returns:
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    # print(data[0])
    imgs = [i[0] for i in data]
    input_ids = [i[1] for i in data]
    masks = [i[2] for i in data]
    indice = [i[3] for i in data]
    
    imgs = torch.stack(imgs).float()
    input_ids = torch.Tensor(np.array(input_ids)).int()
    masks = torch.Tensor(np.array(masks)).long()
    indice = torch.Tensor(np.array(indice)).long()
    
    # print(labels)
    return imgs,input_ids, masks, indice


if __name__ == '__main__':
    args = get_pub_loader_kwargs('dataset/coco/images/val2017/', 'dataset/coco/annos/annotations/captions_val2017.json', num_pub_samples=-1)
    loader = data.DataLoader(**args)

    for batch in loader:
        print(batch)
        break
    pass


