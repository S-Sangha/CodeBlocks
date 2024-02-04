import copy
import os
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# define resnet building blocks

class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(Conv2d(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  Conv2d(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(Conv2d(inchannel, outchannel, 
                                                 kernel_size=1, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        
        out += self.shortcut(x) 
        
        out = F.relu(out) 
        
        return out


    
# define resnet

class ResNet(nn.Module):
    
    def __init__(self, ResidualBlock, num_classes = 16):
        
        super(ResNet, self).__init__()
        
        self.inchannel = 16
        self.conv1 = nn.Sequential(Conv2d(3, 16, kernel_size = 3, stride = 1,
                                            padding = 1, bias = False), 
                                  nn.BatchNorm2d(16), 
                                  nn.ReLU())
        
        self.layer1 = self.make_layer(ResidualBlock, 16, 2, stride = 2)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer5 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer6 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        self.maxpool = MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        
        for stride in strides:
            
            layers.append(block(self.inchannel, channels, stride))
            
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# please do not change the name of this class
def MyResNet():
    return ResNet(ResidualBlock)

class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "dataset/",
        transform=None,
        target_transform=None,
        cache: bool = True,
    ):
        super().__init__()
        self.cache = cache
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.n = 16
        self.label_to_index = {label: index for index, label in enumerate(list(self._get_all_labels()))}
        


        self.dataset = DatasetFolder(
            root=data_dir,
            loader=lambda x: Image.open(x).convert("RGB"),
            extensions=".jpg",  # Adjust the extension based on your image format
            transform=ToTensor(),  # Adjust the transformation based on your needs
        )

        if self.cache:
            from concurrent.futures import ThreadPoolExecutor

            self._images = []
            with ThreadPoolExecutor() as executor:
                self._images = list(
                    tqdm(
                        executor.map(self._load_image, self.dataset.samples),
                        total=len(self.dataset.samples),
                        desc=f"Caching Custom Dataset",
                        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 90),
                    )
                )



    def _load_image(self, sample):
        image_path, label = sample[0], sample[1]

        labels_file_path = os.path.join(os.path.dirname(image_path), 'label.txt')
        
        
        with open(labels_file_path, 'r') as file:
            labels = file.read().splitlines()

      
        img_label = np.zeros(self.n)

        img_label[self.label_to_index[labels[0]]] = 1

        
        img_label = np.array(img_label, dtype=np.int64)


        return image_path, np.array(img_label, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)
    
    def _get_all_labels(self):
        all_labels = ["else","if","while","print","for","x","y","+","=","and","or","not","divisible","1","2","5"]
    
        return all_labels

    def __getitem__(self, idx: int):
        if self.cache:
            image_path, label = self._images[idx]
        else:
            image_path, label = self.dataset.samples[idx]


        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
def run_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer: Optional[Optimizer] = None
):
    training = False if optimizer is None else True
    model.train(training)
    loader = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 60),
    )
    device = torch.device('cpu')
    dtype = torch.float32 

    for t, (x, y) in loader:
        model.train()  # put model to training mode
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        scores = model(x)
        loss = F.cross_entropy(scores, y)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        loss.backward()

        # Update the parameters of the model using the gradients
        optimizer.step()

if __name__ == "__main__":
    dir = "/vol/bitbucket/arj220/ichack/CodeBlocks/dataset/"
    n = 1
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])  
    h, w = 256,256
    aug = {
        "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist()),
            ]
        ),
    }

    datasets = {
        split: CustomDataset(
            data_dir=dir,
            transform=aug[split],
            cache=True,
        )
        for split in ["train", "val"]
    }
    # datasets['test'] = datasets.CLEVRClassification(
    #     root='./', split='test', transform=None
    # )
    datasets["test"] = copy.deepcopy(datasets["val"])

    kwargs = {
        "batch_size": 16,
        "num_workers": os.cpu_count(),  # 4 cores to spare
        "pin_memory": True,
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            **kwargs,
        )
        for split in ["train", "val", "test"]
    }



    model = MyResNet()
    optimizer = optim.Adamax(model.parameters(), lr=0.0025, weight_decay=1e-4)

    # print("\nRunning sanity check...")
    # _ = run_epoch(model, dataloaders["val"],optimizer)
    train_loss = run_epoch(model, dataloaders["train"], optimizer)
    torch.save(model.state_dict(), 'model.pt')
    
    
    for epoch in range(1, 10):
        print("\nEpoch {}:".format(epoch))

        train_loss = run_epoch(model, dataloaders["train"], optimizer)
        torch.save(model.state_dict(), 'model.pt')


   