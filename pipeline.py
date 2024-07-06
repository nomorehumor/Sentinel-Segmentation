from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader
# from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, fcn_resnet101, fcn_resnet50
from model import SegmentationModel, UNet
from training import train, test
from constants import SAVE_DIR, MODELS_DIR
import segmentation_models_pytorch as smp


def load_data(path = SAVE_DIR, batch_size=32):
    
    train = torch.load(path / 'train.pt')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = torch.load(path / 'val.pt')
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test = torch.load(path / 'test.pt')
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_pipeline():
    C = 4
    lr = 0.001
    num_epochs = 20
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')    

    train_loader, val_loader, test_loader = load_data()

    # model = SegmentationModel(num_channels=C)
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )

    # model = UNet(n_class=1)
    model.to(device)

    model = train(model, train_loader=train_loader, val_loader=val_loader, lr=lr, num_epochs=num_epochs, device=device)

    metrics = test(model, test_loader=test_loader, device=device, plot=False)

    save(model, num_epochs, lr, C, metrics)


def save(model, epochs, lr, C, metrics):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    saved_params = {}
    saved_params['learning_rate'] = lr
    saved_params['number_of_epochs'] = epochs
    saved_params['number_of_channels'] = C
    saved_params['pixel_accuracy'] = metrics[0].item()
    saved_params['dice_coefficient'] = metrics[1].item()
    saved_params['precision'] = metrics[2].item()
    saved_params['specificity'] = metrics[3].item()
    saved_params['recall'] = metrics[4].item()
    saved_params['iou'] = metrics[5].item()

    json_txt = json.dumps(saved_params, indent=4)
    json_path = MODELS_DIR / f'params_{timestamp}.json'
    
    with open(json_path, "w") as file: 
        file.write(json_txt)

    model_path = MODELS_DIR / f'model_{timestamp}_{epochs}_{metrics[1]:5.4f}'
    torch.save(model.state_dict(), model_path)  


if __name__== '__main__':
    run_pipeline()