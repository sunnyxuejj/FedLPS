import numpy as np
import os
import torch
import json
import pandas as pd
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


from PIL import Image
import os
import os.path
import pickle
import torch
import torchvision
from typing import Any, Callable, Optional, Tuple


class tiny(torchvision.datasets.VisionDataset):


    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(tiny, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.data: Any = []
        self.targets = []

        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list

        # now load the picked numpy arrays

        root_dir = root + '/tiny-imagenet-200'
        self.identity = root_dir + "/tiny" + str(self.train) + ".pkl"
        if os.path.exists(self.identity):
            tmp_file = open(self.identity, 'rb')
            read_data = pickle.load(tmp_file)
            self.data = read_data[0]
            self.targets = read_data[1]
            tmp_file.close()
        else:
            trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
            trn_file = os.path.join(root_dir, 'train_list.txt')
            tst_file = os.path.join(root_dir, 'val_list.txt')
            with open(trn_file) as f:
                line_list = f.readlines()
                for line in line_list:
                    img, lbl = line.strip().split()
                    trn_img_list.append(img)
                    trn_lbl_list.append(int(lbl))
            with open(tst_file) as f:
                line_list = f.readlines()
                for line in line_list:
                    img, lbl = line.strip().split()
                    tst_img_list.append(img)
                    tst_lbl_list.append(int(lbl))
            self.root_dir = root_dir
            if self.train:
                self.img_list = trn_img_list
                self.label_list = trn_lbl_list

            else:
                self.img_list = tst_img_list
                self.label_list = tst_lbl_list

            self.size = len(self.img_list)

            self.transform = transform
            # if self.train:
            #     tmp = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list,
            #                           transformer=transform)
            # else:
            #     tmp = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list,
            #                          transformer=transform)
            # self.data = tmp.img
            # self.img_id = tmp.img_id
            for index in range(self.size):
                img_name = self.img_list[index % self.size]
            # ********************
                img_path = os.path.join(self.root_dir, img_name)
                img_id = self.label_list[index % self.size]
                img_raw = Image.open(img_path).convert('RGB')
                self.data.append(np.asarray(img_raw))
                self.targets.append(img_id)
                # if index % 1000 == 999:
                #     print('Load PIL images ' + str(self.train) + ': No.', index)

            self.data = np.vstack(self.data).reshape(-1,3,64,64)
            self.data = self.data.transpose((0,2,3,1))
            tmp_file = open(self.identity, 'wb')
            pickle.dump([self.data, self.targets], tmp_file)
            tmp_file.close()


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.targets[index]
        #
        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img)
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        #
        # return img, target
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)
