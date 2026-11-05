# Third-party attribution (modified file).
# Source: https://github.com/ZFancy/DivOE/blob/722c3369e9bf32a870ec6d14f586c0782a787ddd/utils/tiny_imagenet_loader.py
# Upstream license was not found; this material is not assigned the project MIT License.
# Local modification: explanatory sampling comment; dataset behavior is unchanged.
# See ../THIRD_PARTY_NOTICES.md for the unresolved redistribution permission.

import torch
import numpy as np
#from outlier_synthesis import *

class customTinyImageNet(torch.utils.data.Dataset):
    def __init__(self, tiny_imagenet):
        self.tiny_imagenet=tiny_imagenet
        self.offset = 0  # offset index

    def __len__(self):
        return len(self.tiny_imagenet)

    def __getitem__(self, index):
        # Rotating offset lets the trainer vary the exposed outlier sequence.
        index = (index + self.offset) % len(self.tiny_imagenet)

        img,_ = self.tiny_imagenet[index]

        return img, 0  # 0 is the class

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        if self.transform:
            X = self.transform(X)
        y = self.labels[index]

        return X, y