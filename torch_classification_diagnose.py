import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
    def __init__(self, num_classes=2):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_v2_s(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

test_transform = A.Compose([
                            A.Resize(224,224),
                            A.Normalize(),  
                            A.ChannelShuffle(0.1),
                            A.Rotate(limit=20),
                            A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=20, scale_limit=0.2, p=1),
                            A.HorizontalFlip(p=0.5),
                            ToTensorV2()
                            ])

def add_text_to_image(image, text, position=(10, 10), font_size=10):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./godic.ttf", font_size)  # Ensure 'arial.ttf' is available
    draw.text(position, text, fill="red", font=font)
    return image

def efficientnet_inference(img_path,model_path):
    
    custom_labels={0: '정상', 1: '슬개골 탈구'}
    
    test_dataset = CustomDataset(img_path, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    model=torch.load(model_path)
    model.eval()
    
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.to(device)
            model=model.to(device)
            pred = model(imgs)
    
    disease_name=custom_labels[F.softmax(pred, dim=1).argmax().item()]
    confidence=F.softmax(pred, dim=1).max().item()
    
    image=Image.open(img_path)
    res_plotted = add_text_to_image(image, f"{disease_name} ({confidence:.2f})")
    
    return res_plotted,disease_name,confidence