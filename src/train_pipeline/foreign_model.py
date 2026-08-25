# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379

from torchvision import models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import time
from tempfile import TemporaryDirectory
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import custom_data_trans as ct
# import weights_distr as wd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 25

def change_model_arch(input_4dim_flag:bool):
    '''Loads pretrained pytorch AlexNet model and modifies its architecture, returns modified model.'''
    # model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
    # model = models.swin_b(weights = models.Swin_B_Weights.DEFAULT)
    model = models.vgg16_bn(weights = models.VGG16_BN_Weights.DEFAULT)

    print('ORIG', model)
    model.classifier[-1] = nn.Linear(4096, 41, bias=True)
    # model.head = nn.Linear(in_features=1024, out_features=2, bias=True)
    # model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)

    if input_4dim_flag:
        # vgg16/19
        pretrained_weights = model.features[0].weight
        new_featres = nn.Sequential(*list(model.features.children()))
        new_featres[0] = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)
        new_featres[0].weight.data.normal_(0, 0.001)
        new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

        # alexnet
        # pretrained_weights = model.features[0].weight
        # new_featres = nn.Sequential(*list(model.features.children()))
        # new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=2, padding=2)
        # new_featres[0].weight.data.normal_(0, 0.001)
        # new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        # model.classifier[0] = nn.Dropout(p=0.25, inplace=False)

        # swin tranformer
        # pretrained_weights = model.features[0][0].weight
        # new_featres = nn.Sequential(*list(model.features.children()))
        # new_featres[0][0] = nn.Conv2d(4, 128, kernel_size=(4, 4), stride=(4, 4))
        # new_featres[0][0].weight.data.normal_(0, 0.001)
        # new_featres[0][0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        model.features = new_featres
    print(model)
    return model
data_dir = r'.\data\train_val'
image_datasets = {x: ct.my_dataset(os.path.join(data_dir, x)) for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=EPOCHS):
    '''Trains a model, 
    prints the train and validation statistics, 
    saves the weights of the best performing model.'''
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train': model.train()
                else: model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    # print('input processing')
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc)

                elif phase == 'val':
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        torch.save(model.state_dict(), "torch_model_vgg16_step3_fft_lbp.pt")
    return model


# data_dir = input("Directory with train_val data: ")

new_model = change_model_arch(True)
print("Cahged model arch")
new_model = new_model.to(device)
print("Mosel to device", device)


criterion = nn.CrossEntropyLoss()
print("Have criterion")
optimizer_ft = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)
print("Have optimizer")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
print("Have scheuler")
print("Start training")
model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes)

# util()