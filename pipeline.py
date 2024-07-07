from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader
from model import SegmentationModel, UNet
from training import train, test
from constants import SAVE_DIR, MODELS_DIR
import warnings
warnings.filterwarnings("ignore")


def load_data(dataset_name,  batch_size=32):
    print(f'Loading datasets {dataset_name}')

    train = torch.load(SAVE_DIR / f'{dataset_name}_train.pt')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = torch.load(SAVE_DIR / f'{dataset_name}_val.pt')
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test = torch.load(SAVE_DIR / f'{dataset_name}_test.pt')
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_pipeline():
    C = 4
    lr = 0.001
    num_epochs = 30
    augmented = False
    augmentation_type = ''
    patch_size = 32
    batch_size = 32
    dropout_rate = 0
    early_stopping = False

    dataset_name = f'{str(patch_size)}' if not augmented else f'{str(patch_size)}_augmented_{augmentation_type}'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')    

    train_loader, val_loader, test_loader = load_data(dataset_name)

    model = SegmentationModel(num_channels=C, dropout_rate=0)
    # model = smp.Unet(
    #     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=1,                      # model output channels (number of classes in your dataset)
    # )

    # model = UNet(n_class=1)
    model.to(device)

    model = train(model, train_loader=train_loader, val_loader=val_loader, lr=lr, num_epochs=num_epochs, device=device)

    metrics = test(model, test_loader=test_loader, device=device, plot=False)

    save(model, C, patch_size, augmented, augmentation_type, batch_size, dropout_rate, early_stopping, num_epochs, lr,  metrics)


def save(model, C, patch_size, augmented, augmentation_type, batch_size, dropout_rate, early_stopping, epochs, lr, metrics):
    timestamp = datetime.now().strftime('%d%m_%H%M')

    saved_params = {
        'number_of_channels': C,
        'patch_size': patch_size,
        'augmented': augmented,
        'augmentation_type': augmentation_type,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'early_stopping': early_stopping,
        'learning_rate': lr,
        'number_of_epochs': epochs,
        'pixel_accuracy': metrics[0],
        'dice_coefficient': metrics[1],
        'precision': metrics[2],
        'specificity': metrics[3],
        'recall': metrics[4],
        'iou': metrics[5]
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    json_txt = json.dumps(saved_params, indent=4)
    json_path = MODELS_DIR / f'params_{timestamp}.json'
    
    with open(json_path, "w") as file: 
        file.write(json_txt)

    model_path = MODELS_DIR / f'model_{timestamp}_{epochs}_{metrics[1]:5.4f}'
    torch.save(model.state_dict(), model_path)  


if __name__== '__main__':
    run_pipeline()