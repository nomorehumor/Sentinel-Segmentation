import torch.nn as nn
import torch
import torch.optim as optim

from matplotlib import pyplot as plt

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def train(model, train_loader, val_loader, lr, num_epochs, device, early_stopping, lr_scheduling):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_accuracy = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for images, labels in train_loader:
            images, labels = images.float().to(device), labels.float().to(device)
            labels = labels.unsqueeze(1) 

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += calculate_overlap_metrics(outputs, labels)[1]
        
        val_loss, val_accuracy = evaluate(model, criterion, val_loader, device)
        
        if lr_scheduling:
            scheduler.step(val_loss)

        print(f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss/len(train_loader)}, Training Accuracy: {epoch_accuracy/len(train_loader)}, 
              Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}""")

        if early_stopping:    
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return model


def test(model, test_loader, device, plot=False):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0
    test_pixel_accuracy, test_dice, test_precision, test_recall, test_iou = 0, 0, 0, 0, 0

    with torch.no_grad():
        image, label = next(iter(test_loader))
        image, label = image.float().to(device), label.float().to(device)
        label = label.unsqueeze(1) 
        outputs = model(image)
        label_resized=label
        loss = criterion(outputs, label_resized)
        test_loss += loss.item()
        
        metrics = calculate_overlap_metrics(outputs, label_resized)

        test_pixel_accuracy = metrics[0]
        test_dice = metrics[1]
        test_precision = metrics[2]
        test_recall = metrics[3]
        test_iou = metrics[4]
        
        preds = torch.sigmoid(outputs) > 0.5 
    
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_dice}")
    
    if plot:
        plot_results(image.cpu(), label.cpu(), preds.cpu()) 

    return test_pixel_accuracy, test_dice , test_precision, test_recall,  test_iou 


def evaluate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.float().to(device), labels.float().to(device)
            labels = labels.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calculate_overlap_metrics(outputs, labels)[1]       
    return val_loss / len(val_loader), val_accuracy / len(val_loader)


def plot_results(image, label, pred):

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    image = image.squeeze(0).permute(1,2,0).numpy()
    label = label.squeeze(1).squeeze(0).numpy()
    pred = pred.squeeze(1).squeeze(0).numpy()

    axes[0].imshow(image)
    axes[0].set_title('Image of Berlin')
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    
    plt.tight_layout()
    plt.show()


def calculate_overlap_metrics(pred, gt):
    pred = (torch.sigmoid(pred) > 0.5).float()
    eps=1e-5
    output = pred.view(-1, )
    target = gt.view(-1, ).float()

    tp = torch.sum(output * target)
    fp = torch.sum(output * (1 - target))  
    fn = torch.sum((1 - output) * target)  
    tn = torch.sum((1 - output) * (1 - target)) 

    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return pixel_acc, dice, precision, recall, iou