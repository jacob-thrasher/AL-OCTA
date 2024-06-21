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
from utils import train_step, test_step
from collections import OrderedDict
from temp_scaling import ModelWithTemperature

# Note: I am aware this code is garbage, but it works... so...
def compute_uncertainty_scores(logits, method, temperature=1):
    '''
    Computes uncertainty scores, where higher values indicate higher uncertainty

    Args:
        logits: Model outputs (no softmax)
        method: Uncertainty calculation method, where:
                'least' = least confidence 
                'entropy' = entropy sampling
    '''
    assert method in ['least', 'entropy', 'margin', 'ratio'], f'Expexted parameter method to be in [least, entropy, margin, ratio], got {method}'

    # Apply softmax function
    probs = nn.functional.softmax(logits, dim=1)
    n_classes = probs.size()[1]
    max_values = torch.max(probs, dim=1)
    pred_class = max_values.indices

    if method == 'least': # Least confidence
        max_probs = max_values.values
        # s = (1 - max_probs) * (n_classes / (n_classes-1))
        s = 1 - max_probs

    elif method == 'entropy':
        s = -torch.sum(probs * torch.log2(probs), dim=1) / torch.log2(torch.tensor(n_classes))

    elif method == 'margin':
        top2 = torch.topk(probs, 2, dim=1).values
        s = 1 - (top2[:, 0] - top2[:, 1])

    elif method == 'ratio':
        top2 = torch.topk(probs, 2, dim=1).values
        s = -(top2[:, 0] / top2[:, 1])

    return s.tolist(), pred_class.tolist()

def undersample(path, pids):
    '''
    Undersamples datasets by transferring pids from validation set to training set

    Args:
        path: path to load/save csvs
        pids: list of pids to transfer
    '''
    al_train = pd.read_csv(os.path.join(path, 'train.csv'))
    al_valid = pd.read_csv(os.path.join(path, 'valid.csv'))

    transfer_rows = al_valid[al_valid['ID'].isin(pids)]
    al_train = pd.concat([al_train, transfer_rows], ignore_index=True) # Move rows to train set
    al_valid.drop(index=transfer_rows.index, inplace=True)

    assert len(set(al_train['ID'].tolist()).intersection(set(al_valid['ID'].tolist()))) == 0, f'Found overlap in train and validation set!!'

    al_train.to_csv(os.path.join(path, 'train.csv'), index=False)
    al_valid.to_csv(os.path.join(path, 'valid.csv'), index=False)

def oversample(path, least_conf_class):
    '''
    Oversamples datasets
    '''
    class_lookup = {
        '0': 'NORMAL',
        '1': 'AMD',
        '2': 'CNV',
        '3': 'DR'
    }
    disease = class_lookup[str(least_conf_class)]

    al_train = pd.read_csv(os.path.join(path, 'train.csv'))
    subjects = al_train[al_train['Disease'] == disease]
    subjects = subjects[~subjects.duplicated(keep=False)] # Drop already oversampled subjects

    if len(subjects) == 0: return # Do nothing

    selected = subjects.sample()

    al_train = pd.concat([al_train, selected], ignore_index=True)
    al_train.to_csv(os.path.join(path, 'train.csv'), index=False)




def update_splits(valid_path, model_path, n_transfer=2, device='cuda', uncertainty='least'):
    '''
    Performs uncertainty calculations and updates train/validation csvs with most difficult subjects

    Args:
        validpath: path to validation dataset
        model_path: path to model TODO: Refactor to infer model ppath from valid_path
        n_transfer: number of subjects to transfer
        device: operating device
        uncertainty: method for computing uncertainty
    '''

    print('\nUpdating splits....')
    # Load best model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, 4))

    state_dict = torch.load(model_path)
    new_dict = OrderedDict()
    for key in state_dict:
        value = state_dict[key]
        if 'module' in key:
            key = key.replace('module.', '')

        new_dict[key] = value

    model.load_state_dict(new_dict)
    model.to(device)
    model.eval()

    valid_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(valid_path, 'valid.csv'), dim=dim, binary=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # scaled_model = ModelWithTemperature(model)
    # scaled_model.set_temperature(test_dataloader)

    class_conf = {
        '0': {
            'sum_conf': 0,
            'n_samp': 0,
        },
        '1': {
            'sum_conf': 0,
            'n_samp': 0,
        },
        '2': {
            'sum_conf': 0,
            'n_samp': 0,
        },
        '3': {
            'sum_conf': 0,
            'n_samp': 0,
        },
    }

    pid_conf = {}
    for X, y, pid in tqdm(valid_dataloader, disable=False):
        X = X.to(device)
        y = y.tolist()
        pid = pid.tolist()

        out = model(X)

        scores, preds = compute_uncertainty_scores(out, method=uncertainty, temperature=1)

        for c, p, _id, label in zip(preds, scores, pid, y):
            class_conf[str(c)]['sum_conf'] += p
            class_conf[str(c)]['n_samp'] += 1

            if _id not in pid_conf.keys():
                pid_conf[_id] = {
                    'class': -1,
                    'sum_conf': 0,
                    'n_samp': 0,
                    'avg_conf': 0
                }

            pid_conf[_id]['class'] = label
            pid_conf[_id]['sum_conf'] += p
            pid_conf[_id]['n_samp'] += 1



    avg_confidences = []
    for key in class_conf:
        avg_conf = class_conf[key]['sum_conf'] / class_conf[key]['n_samp'] 
        avg_confidences.append(avg_conf)

    least_conf = np.argmax(np.array(avg_confidences)) # Get highest class with *highest* uncertainty score

    for key in pid_conf:
        pid_conf[key]['avg_conf'] = pid_conf[key]['sum_conf'] / pid_conf[key]['n_samp']


    df = pd.DataFrame(pid_conf).T
    least_conf_class = df[df['class'] == least_conf].sort_values(by='avg_conf', ascending=False)

    n_transfer = min(n_transfer, len(least_conf_class))

    # if n_transfer == 0:
    #     # Duplicate one current entry
    #     oversample(root, least_conf)
    # else:
        # Transfer k from valid to train
        # undersample(root, least_conf_class.index[:n_transfer])

    undersample(valid_path, least_conf_class.index[:n_transfer])




####################################

torch.manual_seed(69)

root = '/users/jdt0025/scratch/OCTA_3mm'
exp_name = 'ratio'
dst = os.path.join('figures', exp_name)
uncertainty = 'ratio'

if not os.path.exists(dst):
    # Prep files (altered AL csvs are saved for further analysis if necessary)
    os.mkdir(dst)
    shutil.copy(os.path.join(root, 'AL_train.csv'), os.path.join(dst, f'train.csv'))
    shutil.copy(os.path.join(root, 'AL_valid.csv'), os.path.join(dst, f'valid.csv'))
else:
    raise OSError(f'Directory {dst} already exists')

dim = 299
al_iter = 10
n_transfer = 2
lr = 2e-5
epochs = 5



al_f1s = []
for i in range(al_iter):
    print(f"\n-------------------------")
    print(f"STARTING AL ITERATION {i}")
    print(f"-------------------------")

    os.mkdir(os.path.join(dst, f'iter_{i}'))


    train_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(dst,  f'train.csv'), oversample=False, dim=dim, binary=False)
    test_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'AL_test.csv'), dim=dim, binary=False)

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
    update_splits(dst, f'{dst}/iter_{i}/best_model.pt', device=device, n_transfer=n_transfer, uncertainty=uncertainty)





