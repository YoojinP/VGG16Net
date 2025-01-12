import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class VGG16(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(VGG16, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (224, 224, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (112, 112, 64)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (56, 56, 128)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (28, 28, 256)
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (14, 14, 512)
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # (7, 7, 512)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)  # softamx
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = "./data"
    epoch = 5
    mode = 'train'  # 'test'
    weight_path = 'model_weights.pth'

    if not os.path.exists(path):
        os.mkdir(path)

    transformer = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Resize(224),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(path, train=True, download='True', transform = transformer)
    test_dataset = datasets.CIFAR10(path, train=False, download='True', transform = transformer)
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
    
    test_labels = [y for _, y in test_dataset]
    val_labels = list(range(len(test_dataset))) #[i for i in range(len(test_dataset))]

    for test_index, val_index in split.split(val_labels, test_labels):
        val_dataset = Subset(test_dataset, val_index)
        test_dataset = Subset(test_dataset, test_index)

    y_test = [y for _, y in test_dataset]
    y_val = [y for _, y in val_dataset]

    # 데이터가 카테고리별로 나뉘었는지 확인하는 역할
    # counter_test = collections.Counter(y_test)
    # counter_val = collections.Counter(y_val)
    # print(counter_test)
    # print(counter_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=64)
    # lr_scheduler = optim.lr_scheduler.StepLR


    model = VGG16(in_channel=3, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.0005)
    loss_func = nn.CrossEntropyLoss().to(device)


    train_losses = []
    val_losses = []
    val_acc = []
    train_acc = []
    train_correct =0 
    flag = True

    if mode=='train':
        model.train()
    else:
        model.load_state_dict(torch.load(weight_path))
        flag = False

    for e in range(epoch):
        
        if flag==False:
            break

        optimizer.zero_grad()
        train_loss = 0
        train_correct =0 
        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            img, label = data
            img = img.to(device)
            label = label.to(device)

            y = model(img)
            loss = loss_func(y, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted_idx = torch.max(y.data, dim=1)
            train_correct += (predicted_idx==label).sum().item()

        train_accuracy = 100*(train_correct/len(train_loader.dataset))
        train_acc.append(train_accuracy)
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        print("Train set: ep:{0} loss: {1}, Accuracy: {2}%".format(e+1 ,avg_train_loss, train_accuracy))
        print()

        # 검증 val
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader,0):
                val_x, val_y = data
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                y = model(val_x)
                loss = loss_func(y, val_y)
                val_loss += loss.item()

                values, indices = torch.max(y.data, dim=1)
                correct += (indices==val_y).sum().item()

        val_accuracy = 100 * (correct/len(val_loader.dataset))
        val_acc.append(val_accuracy)
        avg_val_loss = val_loss/len(val_loader.dataset)
        print("Validation set: loss: {0}, Accuracy: {1}%".format(avg_val_loss, val_accuracy))
        print()
        val_losses.append(avg_val_loss)

    torch.save(model.state_dict(), 'model_weights.pth')


    # TEST
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader,0):
            img, label = data
            img = img.to(device)
            label = label.to(device)

            total += label.size(0)
            test_y = model(img)
            _, predicted_indices = torch.max(test_y.data, dim=1)
            correct += (predicted_indices==label).sum().item()

        test_accuracy = 100 * (correct/total)        
        print("Test set: Accuracy: {0}%".format(test_accuracy))
        

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("VGG16 Loss")
    plt.ylabel("Loss")
    plt.xlabel("epoch")

    plt.legend(["Train", "Validation"], loc="best")
    plt.show()
    plt.savefig(fname= "loss.png")

    plt.plot(train_acc)
    plt.plot(val_acc)

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")

    plt.legend(["Train", "Validation"], loc="best")
    plt.show()
    plt.savefig(fname= "acc.png")
