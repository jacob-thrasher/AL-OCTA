import os
import torch
import matplotlib.pyplot as plt
from utils import train_step, test_step
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop, SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchvision import models
from tqdm import tqdm
from data import OCTA500
from network import SimpleCNN

torch.manual_seed(69)

exp_id = 'AutoAug'
dim = 299
model_name = 'inception_v3'

if not os.path.exists(f'figures/{exp_id}'):
    os.mkdir(os.path.join('figures', exp_id))
else:
    raise OSError(f'Directory {exp_id} already exists')

root = 'D:\\Big_Data\\OCTA500\\OCTA\\OCTA_3mm'
train_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'train.csv'), oversample=False, dim=dim, binary=False)
valid_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'valid.csv'), dim=dim, binary=False)
test_dataset = OCTA500(os.path.join(root, 'OCTA'), csvpath=os.path.join(root, 'test.csv'), dim=dim, binary=False)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print(len(train_dataset), len(valid_dataset))

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
# model = models.vit_b_16(models.ViT_B_16_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(nn.Linear(2048, 512), nn.Linear(512, 4))
# model.heads = nn.Sequential(nn.Linear(768, 2))
# model.fc = nn.Linear(512, 2)
# model = SimpleCNN(out_features=2)

device = 'cuda'
print(f'Using {device}')
model.to(device)

weights = torch.tensor([1/127, 1/5, 1/4, 1/23]).to(device)
weights = weights / torch.sum(weights)
loss_fn = CrossEntropyLoss(weight=weights)
# loss_fn = BCEWithLogitsLoss()
optim = Adam(model.parameters(), lr=2e-5)
# optim = RMSprop(model.parameters(), lr=0.0001)
# optim = SGD(model.parameters(), lr=1e-4)


epochs = 10

best_acc = 0
best_f1 = 0
train_losses = []
valid_losses = []
accs = []
f1s = []
for epoch in range(epochs):
    print(f'Epoch: {[epoch]}/{[epochs]}')

    train_loss = train_step(model, train_dataloader, loss_fn, optim, device)
    valid_loss, acc, f1 = test_step(model, test_dataloader, loss_fn, device)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    accs.append(acc)    
    f1s.append(f1)

    if f1 >= best_f1:
        best_acc = acc
        best_f1 = f1
        torch.save(model.state_dict(), f'figures/{exp_id}/best_model.pt')


    print('Train loss:', train_loss)
    print('Valid loss:', valid_loss)
    print('Accuracy:', acc)
    print('F1:', f1)
    print('\nBest Acc:', best_acc)
    print('Best F1:', best_f1)

    
    plt.plot(train_losses, color='blue', label='Train')
    plt.plot(valid_losses, color='orange', label='Valid')
    plt.title('Train and validation loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'figures/{exp_id}/loss.png')
    plt.close()

    plt.plot(accs, color='green', label='Accuracy')
    plt.plot(f1s, color='purple', label='F1-score')
    plt.title('Metrics over time')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'figures/{exp_id}/acc_f1.png')
    plt.close()