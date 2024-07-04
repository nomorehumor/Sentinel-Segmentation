import torchvision.transforms as T
from torchvision.transforms import functional as TF

patch_size = 32

augmentations = {
    "rotation": T.RandomRotation(30),
    "reflection": T.RandomHorizontalFlip(1),
    "zoom": T.RandomResizedCrop(size=patch_size, scale=(0.6, 1.0)),
    #"modulated_noise": lambda img: img + torch.randn_like(img) * 0.1,
    "affine": T.RandomAffine((50, 60))
}

compound_augmentations = {
    "rotate_reflect": T.Compose([augmentations["rotation"], augmentations["reflection"]]),
    "rotate_zoom": T.Compose([augmentations["rotation"], augmentations["zoom"]]),
    "reflect_zoom": T.Compose([augmentations["reflection"], augmentations["zoom"]]),
    "rotate_reflect_affine": T.Compose([augmentations["rotation"], augmentations["reflection"], augmentations["affine"]]),
    "all": T.Compose([augmentations["rotation"], augmentations["reflection"], augmentations["zoom"], augmentations["affine"]])
}

def apply_transform(data, transform):
    x, y = data
    x = TF.to_pil_image(x)
    x = transform(x)
    x = TF.to_tensor(x)
    return x, y

def augment_dataset(dataset, augmentations, compound_augmentations):
    augmented_datasets = {}
    for name, transform in augmentations.items():
        augmented_datasets[name] = [(apply_transform(data, transform)) for data in dataset]
    for name, transform in compound_augmentations.items():
        augmented_datasets[name] = [(apply_transform(data, transform)) for data in dataset]
    return augmented_datasets