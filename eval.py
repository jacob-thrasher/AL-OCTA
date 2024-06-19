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

path = 'figures/AugMix_oversample2'
average = 'macro'
dim = 299
pretrained_model_path = os.path.join(path, 'best_model.pt')


root = '/users/jdt0025/scratch/OCTA_3mm'

test_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'test.csv'), dim=dim, binary=False)

batch_size = 64
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
_, acc, f1 = test_step(model, test_dataloader, loss_fn, device, average=average)
print(acc, f1)

# create_confusion_matix(model, test_dataloader, ['Normal', 'AMD', 'CNV', 'DR'], os.path.join(path, 'cm.png'), device, normalize=None)
# create_confusion_matix(model, test_dataloader, ['Normal', 'AMD', 'CNV', 'DR'], os.path.join(path, 'cm_recall.png'), device, normalize='true')
# create_confusion_matix(model, test_dataloader, ['Normal', 'AMD', 'CNV', 'DR'], os.path.join(path, 'cm_prec.png'), device, normalize='pred')

# plt.hist(arr, bins=100)
# plt.show()

# img, _ = dataset[1]
# print(img.size())