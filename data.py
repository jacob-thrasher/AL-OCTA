import os
import torch
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

class OCTA500(Dataset):
    def __init__(self, root, csvpath, dim=224, random_crop_ratio=.75, oversample=False, split=None, binary=False):
        '''
        Args:
            root (path): path to image root
            csvpath (path): path to csv file containing label information

        Keyword Args:
            dim (int): Image dimension
            random_crop_ratio (float): Ratio of dim arg to randomly crop image
        '''
        self.root = root
        self.df = pd.read_csv(csvpath)
        self.df = self.df.dropna(how='all')

        if split:
            if split == 'NORMAL':
                self.df = self.df[self.df['Disease'] == 'NORMAL']
            else:
                self.df = self.df[self.df['Disease'] != 'NORMAL']

        # There is def a better way to do this but I can't be bothered rn
        self.elements = []
        for i, row in self.df.iterrows():
            entries = []
            for j in range(1, 305):
                entries.append((int(row.ID), j))

            # If oversample, double diseased entries in list
            if oversample and row.Disease != 'NORMAL': 
                entries = entries * 2
            self.elements += entries
                
        self.label_map = {
            'NORMAL': 0,
            'AMD': 1,
            'CNV': 2,
            'DR': 3
        }
        self.binary = binary

        self.process = T.Compose([
            T.ToTensor(),
            T.Normalize((0), (1)),
            T.CenterCrop((dim, dim)),
            # T.RandomCrop((int(dim*random_crop_ratio)), (int(dim*random_crop_ratio))),
            # T.Resize((dim, dim)),
            T.RandomHorizontalFlip(),
        ])
    
    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, idx):
        _id, img_idx = self.elements[idx]
        img = Image.open(os.path.join(self.root, str(_id), f'{img_idx}.bmp')).convert('L')

        disease = self.df[self.df['ID'] == _id].iloc[0].Disease
        if self.binary:
            label = 0 if disease == 'NORMAL' else 1
        else:
            label = self.label_map[disease]

        pid = self.df[self.df['ID'] == _id].iloc[0].ID

        return self.process(img).repeat(3, 1, 1), label, pid
        # return img, label