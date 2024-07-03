from datetime import datetime
import torch.nn as nn
import torch
import torch.optim as optim

from matplotlib import pyplot as plt


def train(model, train_loader, val_loader, lr, num_epochs, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for images, labels in train_loader:
            images, labels = images.float().to(device), labels.float().to(device)
            labels = labels.unsqueeze(1)  # add channel dim

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy(outputs, labels)
        
        val_loss, val_accuracy = evaluate(model, criterion, val_loader, device)
        
        print(f"""Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss/len(train_loader)}, Training Accuracy: {epoch_accuracy/len(train_loader)}, 
              Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}""")
        
    model_path = 'model_{}_{}'.format(timestamp, val_accuracy)
    torch.save(model.state_dict(), model_path)  


def test(model, test_loader, device, num_images=5, plot=True):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0
    test_accuracy = 0
    images_list, labels_list, preds_list = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.float().to(device), labels.float().to(device)
            labels = labels.unsqueeze(1)  
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_accuracy += accuracy(outputs, labels)
            
            preds = torch.sigmoid(outputs) > 0.5 
            
            images_list.append(images.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())
            
            if len(images_list) * images.size(0) >= num_images:
                break
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_accuracy / len(test_loader)
    
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
    
    if plot:
        images_list = torch.cat(images_list)[:num_images]
        labels_list = torch.cat(labels_list)[:num_images]
        preds_list = torch.cat(preds_list)[:num_images]
        plot_results(num_images, images_list, labels_list, preds_list)        


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
            val_accuracy += accuracy(outputs, labels)
    return val_loss / len(val_loader), val_accuracy / len(val_loader)


def accuracy(outputs, labels, threshold=0.5):
    preds = torch.sigmoid(outputs) > threshold 
    correct = (preds == labels).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


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