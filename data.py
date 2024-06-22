import os
import torch
import pandas as pd
import torchvision.transforms as T
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset

class OCTA500(Dataset):
    def __init__(self, root, csvpath, dim=224, oversample=False, split=None, binary=False, df_style='subject'):
        '''
        Args:
            root (path): path to image root
            csvpath (path): path to csv file containing label information

        Keyword Args:
            dim        (int) : Image dimension, Default: 224
            oversample (bool): Oversample minority classes? Default: False
            split      (str) : Load only selected disease, Default: None
                                Should be one of: [NORMAL, CNV, DR, AMD]
            binary     (bool): Lump all diseased eyes to one class for binary classification, Default: False
            df_style   (str) : Inidcates which dataframe is being loaded. Subject style only contains subject
                                level information, while Instance style has one row per image.
                                Instance style dfs should contain an additional "Instance" column which contains
                                the image ID associated with the subject ID
                                Should be one of [subject, instance]
        '''
        self.root = root
        self.df_style = df_style
        self.df = pd.read_csv(csvpath)
        self.df = self.df.dropna(how='all')

        if split:
            self.df = self.df[self.df['Disease'] == split]


        # Construct list of (ID, instance) tuples from df.
        if df_style == 'subject': # TODO: Clean subject style loading
            self.elements = []
            for i, row in self.df.iterrows():
                entries = []
                for j in range(1, 305):
                    entries.append((int(row.ID), j))

                # If oversample, double diseased entries in list
                if oversample and row.Disease != 'NORMAL': 
                    entries = entries * 2
                self.elements += entries
        elif df_style == 'instance':
            self.elements = list(zip(self.df['ID'].tolist(), self.df['Instance'].tolist()))
                
        self.label_map = {
            'NORMAL': 0,
            'AMD': 1,
            'CNV': 2,
            'DR': 3
        }
        self.binary = binary

        self.process = v2.Compose([
            T.ToTensor(),
            T.Normalize((0), (1)),
            T.CenterCrop((dim, dim)),
            T.RandomHorizontalFlip(),
        ])

        # self.process = v2.Compose([
        #                     v2.ToImage(),
        #                     v2.ToDtype(torch.float32, scale=True),
        #                     v2.RGB(),
        #                     v2.CenterCrop(dim),
        #                     v2.AutoAugment()
        #                 ])
    
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


        if self.df_style == 'subject':
            pid = self.df[self.df['ID'] == _id].iloc[0].ID 
        elif self.df_style == 'instance':
            pid = str(_id) + 'S' + str(img_idx)

        return self.process(img).repeat(3, 1, 1), label, pid
