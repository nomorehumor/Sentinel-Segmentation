from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader
from model import SegmentationModel
import segmentation_models_pytorch as smp
from training import train, test
from constants import TRAINING_DATASET_DIR, MODELS_DIR, PARAMS_DIR, RESULTS_DIR
import warnings
warnings.filterwarnings("ignore")


def load_data(dataset_name, batch_size=32):
    print(f'Loading datasets {dataset_name}_train _val _test')

    train = torch.load(TRAINING_DATASET_DIR / f'{dataset_name}_train.pt')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = torch.load(TRAINING_DATASET_DIR / f'{dataset_name}_val.pt')
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test = torch.load(TRAINING_DATASET_DIR / f'{dataset_name}_test.pt')
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def run_pipeline():
    model_type = 'simple'
    C = 4
    lr = 0.001
    num_epochs = 30
    augmented = False
    augmentation_type = ''
    patch_size = 64
    batch_size = 32
    dropout_rate = 0.2
    early_stopping = True
    lr_scheduling = True
    # dataset_name = "64_rotation"

    dataset_name = f'{str(patch_size)}' if not augmented else f'{str(patch_size)}_augmented_{augmentation_type}'
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')    

    train_loader, val_loader, test_loader = load_data(dataset_name)

    if model_type == 'unet':
        model = smp.Unet(
        encoder_name="resnet34",       
        encoder_weights="imagenet",     
        in_channels=C,                 
        classes=1,                     
        )
    else:
        model = SegmentationModel(num_channels=C, dropout_rate=dropout_rate)

    model.to(device)

    model = train(model, 
                  train_loader=train_loader, 
                  val_loader=val_loader, 
                  lr=lr, 
                  num_epochs=num_epochs, 
                  device=device, 
                  early_stopping=early_stopping,
                  lr_scheduling=lr_scheduling)

    metrics = test(model, test_loader=test_loader, device=device, plot=True)

    save(model_type, model, C, patch_size, augmented, augmentation_type, batch_size, dropout_rate, \
          early_stopping, lr_scheduling, num_epochs, lr,  metrics)


def save(model_type, model, C, patch_size, augmented, augmentation_type, batch_size, dropout_rate, \
          early_stopping, lr_scheduling, epochs, lr, metrics):
    
    timestamp = datetime.now().strftime('%d%m_%H%M')

    saved_params = {
        'number_of_channels': C,
        'patch_size': patch_size,
        'augmented': augmented,
        'augmentation_type': augmentation_type,
        'batch_size': batch_size,
        'model_type': model_type,
        'learning_rate': lr,
        'number_of_epochs': epochs,
        'dropout_rate': dropout_rate,
        'early_stopping': early_stopping,
        'lr_scheduling': lr_scheduling,
        'pixel_accuracy': metrics[0].item(),
        'dice_coefficient': metrics[1].item(),
        'precision': metrics[2].item(),
        'recall': metrics[3].item(),
        'iou': metrics[4].item()
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    json_txt = json.dumps(saved_params, indent=4)
    json_path = PARAMS_DIR / f'{model_type}_{timestamp}.json'
    
    with open(json_path, "w") as file: 
        file.write(json_txt)

    model_path = MODELS_DIR / f'{model_type}_{timestamp}_{epochs}_{metrics[1]:5.4f}'
    torch.save(model.state_dict(), model_path)  


if __name__== '__main__':
    run_pipeline()