import torch
import torchmetrics.functional as tmf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm


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
