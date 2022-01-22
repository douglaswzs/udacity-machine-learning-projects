import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import json

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    return test_loss
    pass

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss=0
    correct=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = criterion(output, target)
        running_loss+=loss.item()*data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            running_loss/len(train_loader.dataset), correct, len(train_loader.dataset), 100.0 * correct / len(train_loader.dataset)
        )
    )
    pass
    
def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model
    pass

def create_data_loaders(data, batch_size, test_batch_size, num_cpus, num_gpus):
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}
            
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    worker_count = 4
    if num_gpus > 0:
        worker_count = min(num_gpus, worker_count)
    elif num_cpus > 0:
        worker_count = min(num_cpus, worker_count)
    
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data,'train'), transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(data,'valid'), transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=worker_count)
    test_loader = torch.utils.data.DataLoader(test_dataset, test_batch_size, num_workers=worker_count)
    
    return train_loader, test_loader
    pass

def main(args):
    model=net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model=model.to(device)
    
    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size, args.test_batch_size, args.num_cpus, args.num_gpus)
    
    '''
    We use the CrossEntropyLoss loss function as we are categorising across 133 categories.
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    print("\nHyperparameters [Learning Rate {:.4e}, eps {:.4e}, weight decay {:.4e}\n".format(args.lr, args.eps, args.weight_decay))
    
    for epoch in range(1, args.epochs + 1):
        print("\nEpoch: {}\nTraining".format(epoch))
        train(model, train_loader, loss_criterion, optimizer, device)
        print("\nEpoch: {}\nTesting".format(epoch))
        loss = test(model, test_loader, loss_criterion, device)
    
    '''
    Save the trained model
    '''
    torch.save(model, os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    '''
    3 hyperparameters used for the AdamW optimiser (learning rate, eps, weight decay). The defaults are as in pytorch
    '''
    parser.add_argument(
        "--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, metavar="EPS", help="eps (default: 1e-8)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 1e-2)"
    )
    

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])

    args=parser.parse_args()
    
    main(args)