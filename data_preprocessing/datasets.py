import logging
from typing import Optional, Callable

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
import os
import pandas as pd
# from medmnist import ChestMNIST
# import hub

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


# class ChestMINIST_truncated(ChestMNIST):
class ChestMINIST_truncated:
    pass
    
#     def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

#         self.split = 'train' if train else 'test'
#         super(ChestMINIST_truncated,self).__init__(root=root, split=self.split, 
#                                 transform=transform, target_transform=target_transform, download=download)
#         self.dataidxs = dataidxs
#         self.train = train

#         if self.dataidxs is not None:
#             self.imgs = self.imgs[self.dataidxs]
#             self.labels = self.labels[self.dataidxs]

#         self.data = self.imgs
#         self.target = self.labels

#     def truncate_channel(self, index):
#         for i in range(index.shape[0]):
#             gs_index = index[i]
#             self.imgs[gs_index, :, :, 1] = 0.0
#             self.imgs[gs_index, :, :, 2] = 0.0

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.imgs[index], self.labels[index]

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)        

class ChestXray14(VisionDataset):
    
    def __init__(self, root: str, train=True, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, dataidxs=None, download=True):
        
        super().__init__(root, transforms, transform, target_transform)
        self.dataidxs = dataidxs
        # Get splits
        split_path = os.path.join(root, 'train_val_list.txt') if train else os.path.join(root, 'test_list.txt')
        with open(split_path) as f:
            splits = f.readlines()
        splits = [s.strip('\n') for s in splits]
        self.splits = splits

        # Get annos
        csv_path = os.path.join(root, "Data_Entry_2017.csv")
        annos = pd.read_csv(csv_path).set_index('Image Index')

        # add file path
        annos['folder'] = [os.path.join(root, self._idx_to_folder(i)) for i in range(len(annos))]

        self.annos = annos['Finding Labels'][splits].reset_index().values.tolist()
        self.path = annos['folder'][splits].reset_index().values.tolist()

        self.path = np.array(self.path)

        for info in self.annos:
            info[-1] = info[-1].partition('|')[0]
        # for k,v in self.annos.items(): self.annos[k] = v.partition('|')[0]

        split_path = os.path.join(root, 'train_val_list.txt') if train else os.path.join(root, 'test_list.txt')
        with open(split_path) as f:
            splits = f.readlines()

        self.loader = default_loader

        self.classes = sorted(set([label for _, label in self.annos]))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.data = np.array([file_name for file_name, label in self.annos])
        self.target = np.array([self.class_to_idx[label] for file_name, label in self.annos])



        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]
            self.path = self.path[self.dataidxs]

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])

    def _idx_to_folder(self,idx):
        nums = [4999] + [10000]*10 + [7121]

        base = os.path.join('images_%03d','images')
        for i in range(1,len(nums)+1):
            ceiling =  sum(nums[:i])
            if idx < ceiling:
                return base % i

    def __getitem__(self, index: int):
        file_name = self.data[index]
        label = self.target[index]
        name, path = self.path[index]

        assert file_name == name, "Data loaded incorrectly!"

        file_path = os.path.join(path,name)

        sample = self.loader(file_path)

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform

        return sample, label



    def __len__(self):
        return len(self.data)





class CIFAR_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        if "cifar100" or "cifar-100" in self.root:
            cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        else:
            cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)


        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

# Reduce sample number
class CifarReduced(CIFAR_truncated):
    def __init__(self, root, sample_num, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()
        total_num = len(self.data)
        if total_num > sample_num and self.train:
            self.dataidxs = np.random.choice(total_num, sample_num)
            self.data, self.target = self.__build_truncated_dataset__()

        



# Imagenet
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            imagefolder_obj = ImageFolder(self.root+'/train', self.transform)
        else:
            imagefolder_obj = ImageFolder(self.root+'/val', self.transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
        self.target = self.samples[:,1].astype(np.int64)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

class ImageFolderTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, download=False):
        if train:
            root = root+'/train'
        else:
            root = root+'/test'
        super(ImageFolderTruncated, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                   transform=transform,
                                                   target_transform=target_transform,
                                                   is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.dataidxs = dataidxs

        ### we need to fetch training labels out here:
        self.target = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            # self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target