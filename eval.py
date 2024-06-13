import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from data import OCTA500
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils import create_confusion_matix, test_step
from collections import OrderedDict


torch.manual_seed(69)

dim = 299
pretrained_model_path = 'figures/inception_v3_4-class/best_model.pt'


root = 'D:\\Big_Data\\OCTA500\\OCTA\\OCTA_3mm'
train_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'train.csv'), oversample=False, dim=dim, binary=False)
valid_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'valid.csv'), dim=dim, binary=False)
test_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'test.csv'), dim=dim, binary=False)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
# model = models.vit_b_16(models.ViT_B_16_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, 4))

state_dict = torch.load(pretrained_model_path)
new_dict = OrderedDict()
for key in state_dict:
    value = state_dict[key]
    if 'module' in key:
        key = key.replace('module.', '')

    new_dict[key] = value

model.load_state_dict(new_dict)
model.eval()


device = 'cuda'
print(f'Using {device}')
model.to(device)

loss_fn = nn.CrossEntropyLoss()
_, acc, f1 = test_step(model, test_dataloader, loss_fn, device)
print(acc, f1)

create_confusion_matix(model, test_dataloader, ['Normal', 'AMD', 'CNV', 'DR'],'figures/inception_v3_4-class/cm.png', device)

# plt.hist(arr, bins=100)
# plt.show()

# img, _ = dataset[1]
# print(img.size())