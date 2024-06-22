import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, RMSprop, SGD
from data import OCTA500
from torch import nn
from torch.utils.data import DataLoader
from utils import train_step, test_step, update_splits_subject, update_splits_instance
from collections import OrderedDict
from temp_scaling import ModelWithTemperature


torch.manual_seed(69)

root = 'D:\\Big_Data\\OCTA500\\OCTA\\OCTA_3mm'
exp_name = 'LC_instance2'
dst = os.path.join('figures', exp_name)
uncertainty = 'least'

if not os.path.exists(dst):
    # Prep files (altered AL csvs are saved for further analysis if necessary)
    os.mkdir(dst)
    shutil.copy(os.path.join(root, 'AL_train_instance.csv'), os.path.join(dst, f'train.csv'))
    shutil.copy(os.path.join(root, 'AL_valid_instance.csv'), os.path.join(dst, f'valid.csv'))
else:
    raise OSError(f'Directory {dst} already exists')

dim = 299
al_iter = 20
n_transfer = 608
lr = 2e-5
epochs = 5



al_f1s = []
for i in range(al_iter):
    print(f"\n-------------------------")
    print(f"STARTING AL ITERATION {i}")
    print(f"-------------------------")

    os.mkdir(os.path.join(dst, f'iter_{i}'))


    train_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(dst,  f'train.csv'), oversample=False, dim=dim, binary=False, df_style='instance')
    test_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'AL_test_instance.csv'), dim=dim, binary=False, df_style='instance')

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    print(len(train_dataset), len(test_dataset))


    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, 4))

    device = 'cuda'
    print(f'Using {device}')
    model.to(device)

    loss_fn = CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)

    best_acc = 0
    best_f1 = 0
    train_losses = []
    valid_losses = []
    accs = []
    f1s = []
    for epoch in range(epochs):
        print(f'Epoch: {[epoch]}/{[epochs]}')

        train_loss = train_step(model, train_dataloader, loss_fn, optim, device)
        valid_loss, acc, f1 = test_step(model, test_dataloader, loss_fn, device, average='macro')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        accs.append(acc)    
        f1s.append(f1)

        if f1 >= best_f1:
            best_acc = acc
            best_f1 = f1
            torch.save(model.state_dict(), f'{dst}/iter_{i}/best_model.pt')


        print('Train loss:', train_loss)
        print('Valid loss:', valid_loss)
        print('Accuracy:', acc)
        print('F1:', f1)


        # Plotting
        plt.plot(train_losses, color='blue', label='Train')
        plt.plot(valid_losses, color='orange', label='Valid')
        plt.title('Train and validation loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'{dst}/iter_{i}/loss.png')
        plt.close()

        plt.plot(accs, color='green', label='Accuracy')
        plt.plot(f1s, color='purple', label='F1-score')
        plt.title('Metrics over time')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'{dst}/iter_{i}/acc_f1.png')
        plt.close()

    # Post training metrics and plotting
    print('-----------')
    print("Best Acc:", best_acc)
    print("Best F1 :", best_f1)
    al_f1s.append(best_f1)

    plt.plot(al_f1s, color='red', label='f1')
    plt.title('Best f1 over active learning iterations')
    plt.xlabel('AL iter')
    plt.legend()
    plt.savefig(f'{dst}/f1s.png')
    plt.close()

    # Active Learning step
    valid_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(dst, 'valid.csv'), dim=dim, binary=False, df_style='instance')
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    # update_splits_subject(dst, f'{dst}/iter_{i}/best_model.pt', device=device, n_transfer=n_transfer, uncertainty=uncertainty)
    update_splits_instance(dst, valid_dataloader, f'{dst}/iter_{i}/best_model.pt', device=device, n_transfer=n_transfer, uncertainty=uncertainty)





