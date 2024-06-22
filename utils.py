import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchmetrics.functional as tmf
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
from collections import OrderedDict
from data import OCTA500


def train_step(model, dataloader, loss_fn, optim, device):
    model.train()

    running_loss = 0
    for X, y, _ in tqdm(dataloader, disable=False):
        X = X.to(device)
        y = y.to(device)


        pred = model(X)[0]#.squeeze() # Resnet

        loss = loss_fn(pred, y)
        loss.backward()
        running_loss += loss.item()
        optim.step()
        optim.zero_grad()


    return running_loss / len(dataloader)

def test_step(model, dataloader, loss_fn, device, average='micro'):
    model.eval()

    running_loss = 0
    acc = 0
    f1 = 0
    for X, y, _ in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)#.squeeze() # Resnet

        loss = loss_fn(pred, y)
        running_loss += loss.item()

        pred = torch.argmax(pred, dim=1)
        acc += tmf.accuracy(pred, y, task='multiclass', num_classes=4, average=average).item()
        f1 += tmf.f1_score(pred, y, task='multiclass', num_classes=4, average=average).item()

    return running_loss / len(dataloader), acc / len(dataloader), f1 / len(dataloader)

def plot_confusion_matrix(pred, labels, classes, normalize=None):
    cm = confusion_matrix(labels, pred, normalize=normalize)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix on test set',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.3f' if normalize is not None else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if normalize:
        title = 'Recall' if normalize == 'true' else 'Precision'
    else:
        title = 'Unnormalized'
    plt.title(f"Confusion matrix ({title})")
    return fig, ax

def create_confusion_matix(model, dataloader, class_labels, dst, device, normalize=None):
    print("Generating confusion matrix...")
    all_pred = []
    all_labels = []

    model.eval()
    for i, (X, y, _) in enumerate(tqdm(dataloader)):
        X = X.to(device)
    
        pred = model(X)
        # pred = torch.round(pred.detach().cpu()).squeeze()
        pred = torch.argmax(pred, dim=1)
        pred = [int(x.item()) for x in list(pred)]
        all_pred += pred
        all_labels += y.tolist()

    fig, ax = plot_confusion_matrix(all_pred, all_labels, classes=class_labels, normalize=normalize)
    fig.savefig(dst)



#########################
#                       #
# ACTIVE LEARNING UTILS #
#                       #
#########################

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

def undersample(path, pids, col='ID'):
    '''
    Undersamples datasets by transferring pids from validation set to training set

    Args:
        path: path to load/save csvs
        pids: list of pids to transfer
    
    Keyword Args:
        col: column to sample from
    '''
    al_train = pd.read_csv(os.path.join(path, 'train.csv'))
    al_valid = pd.read_csv(os.path.join(path, 'valid.csv'))

    transfer_rows = al_valid[al_valid[col].isin(pids)]
    al_train = pd.concat([al_train, transfer_rows], ignore_index=True) # Move rows to train set
    al_valid.drop(index=transfer_rows.index, inplace=True)

    assert len(set(al_train[col].tolist()).intersection(set(al_valid[col].tolist()))) == 0, f'Found overlap in train and validation set!!'

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




def update_splits_subject(data_path, valid_dataloader, model_path, n_transfer=2, device='cuda', uncertainty='least'):
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

    undersample(data_path, least_conf_class.index[:n_transfer])


# TODO: I can definitely restructure update_splits_instance to fit into this function
def update_splits_instance(data_path, valid_dataloader, model_path, n_transfer=2, device='cuda', uncertainty='least'):
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

    # scaled_model = ModelWithTemperature(model)
    # scaled_model.set_temperature(test_dataloader)

    
    uncertainty_df = pd.DataFrame(columns=['ID', 'Disease', 'Pred', 'Uncertainty'])
    for X, y, pid in tqdm(valid_dataloader, disable=False):
        X = X.to(device)
        y = y.tolist()
        pid = pid

        out = model(X)

        scores, preds = compute_uncertainty_scores(out, method=uncertainty, temperature=1)

        # TODO: Get rid of this for loop
        for p, s, _id, label in zip(preds, scores, pid, y):
            uncertainty_df.loc[len(uncertainty_df)] = [_id, label, p, s]


    uncertainty_df.sort_values(by='Uncertainty', ascending=False, inplace=True)

    n_transfer = min(n_transfer, len(uncertainty_df))

    undersample(data_path, uncertainty_df['ID'][:n_transfer].tolist(), col='Img_ID')