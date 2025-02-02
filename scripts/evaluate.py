import torch
from torch import nn
import numpy as np


# Model, loss function, and device
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_attention_maps = []

    with torch.no_grad():
        for images, one_hot_labels in test_loader:
            images = images.to(device)
            labels = torch.argmax(one_hot_labels, dim=1).to(device)  # Convert one-hot to class indices
            
            outputs, attention_maps = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect attention maps
            for layer_idx, attn_map in enumerate(attention_maps):
                if len(all_attention_maps) <= layer_idx:
                    all_attention_maps.append([])
                all_attention_maps[layer_idx].append(attn_map.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # Convert lists of numpy arrays to numpy arrays
    all_attention_maps = [np.concatenate(attn_maps, axis=0) for attn_maps in all_attention_maps]
    
    return test_loss, test_accuracy, all_attention_maps

