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
    test_pixel_accuracy, test_dice, test_precision, test_specificity, test_recall, test_iou = 0, 0, 0, 0, 0, 0
    images_list, labels_list, preds_list = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.float().to(device), labels.float().to(device)
            labels = labels.unsqueeze(1) 
            outputs = model(images)
            labels_resized = F.interpolate(labels, size=outputs.shape[2:], mode='nearest')
            loss = criterion(outputs, labels_resized)
            test_loss += loss.item()
            
            metrics = calculate_overlap_metrics(outputs, labels_resized)

            test_pixel_accuracy += metrics[0]
            test_dice += metrics[1]
            test_precision += metrics[2]
            test_specificity += metrics[3]
            test_recall += metrics[4]
            test_iou += metrics[5]
            
            preds = torch.sigmoid(outputs) > 0.5 
            
            images_list.append(images.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)
    
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_dice}")
    
    if plot:
        images_list = torch.cat(images_list)
        labels_list = torch.cat(labels_list)
        preds_list = torch.cat(preds_list)
        plot_results(images_list, labels_list, preds_list) 

    return test_pixel_accuracy / len(test_loader), test_dice / len(test_loader), test_precision / len(test_loader), \
    test_specificity / len(test_loader), test_recall / len(test_loader), test_iou / len(test_loader)


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


def plot_results(num_images, images_list, labels_list, preds_list):

    _, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    for i in range(num_images):
        img = images_list[i].permute(1, 2, 0) 
        label = labels_list[i].squeeze(0)
        pred = preds_list[i].squeeze(0)

        if num_images == 1:
            axes[0].imshow(img.numpy())
            axes[0].set_title('Image')
            axes[1].imshow(label.numpy(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[2].imshow(pred.numpy(), cmap='gray')
            axes[2].set_title('Prediction')
        else:
            axes[i, 0].imshow(img.numpy())
            axes[i, 0].set_title('Image')
            axes[i, 1].imshow(label.numpy(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 2].imshow(pred.numpy(), cmap='gray')
            axes[i, 2].set_title('Prediction')
    
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