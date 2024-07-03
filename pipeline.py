import torch
from torch.utils.data import DataLoader

from model import SegmentationModel
from training import train, test
from pathlib import Path

DATASET_DIR = Path("datasets")
SAVE_DIR = DATASET_DIR / "training"


def load_data(path = SAVE_DIR, batch_size=32):
    
    train = torch.load(path / 'train.pt')
    train_loader = DataLoader(train, batch_size=batch_size)
    val = torch.load(path / 'val.pt')
    val_loader = DataLoader(val, batch_size=batch_size)
    test = torch.load(path / 'test.pt')
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def run_pipeline():
    C = 3
    lr = 0.001
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = load_data()

    model = SegmentationModel(num_channels=C)
    model.to(device)

    train(model, train_loader=train_loader, val_loader=val_loader, lr=lr, num_epochs=num_epochs, device=device)

    test(model, test_loader=test_loader, device=device, plot=False)


if __name__== '__main__':
    run_pipeline()