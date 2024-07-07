import random
from typing import Dict, Tuple
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torch.utils.data import TensorDataset
import torch

from constants import TRAIN_PATCH_SIZE 


augmentations = {
    "rotation": T.RandomRotation(30),
    "reflection": T.RandomHorizontalFlip(1),
    "zoom": T.RandomResizedCrop(size=TRAIN_PATCH_SIZE, scale=(0.6, 1.0)),
    #"modulated_noise": lambda img: img + torch.randn_like(img) * 0.1,
    "affine": T.RandomAffine((50, 60))
}

compound_augmentations = {
    "rotate_reflect": T.Compose([augmentations["rotation"], augmentations["reflection"]]),
    "reflect_affine": T.Compose([augmentations["reflection"], augmentations["affine"]]),
    "rotate_zoom": T.Compose([augmentations["rotation"], augmentations["zoom"]]),
    # "reflect_zoom": T.Compose([augmentations["reflection"], augmentations["zoom"]]),
    # "rotate_reflect_affine": T.Compose([augmentations["rotation"], augmentations["reflection"], augmentations["affine"]]),
    "all": T.Compose([augmentations["rotation"], augmentations["reflection"], augmentations["zoom"], augmentations["affine"]])
}

def apply_transform(data: Tuple[torch.Tensor, torch.Tensor], transform) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = data    
    combined = torch.cat([x, y.unsqueeze(0)], dim=0)
    combined = transform(combined)
    
    return combined[:4], combined[4].squeeze(0)

def augment_dataset(dataset: TensorDataset) -> Dict[str, TensorDataset]:
    augmented_datasets = {} # dict of datasets
    for name, transform in compound_augmentations.items():
        augmented_datasets[name] = [(apply_transform(data, transform)) for data in dataset]
    return augmented_datasets