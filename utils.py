import torch
import numpy as np


def save_model(path, model, optimizer, scheduler, train_history, val_history):
    torch.save({
        'epoch': len(train_history),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'val_history': val_history
    }, path)
    print('successfully saved')


def load_model(path, model, optimizer, scheduler=None, train_history=None, val_history=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if train_history:
        train_history = checkpoint['train_history']
    if val_history:
        val_history = checkpoint['val_history']
    print('successfully loaded')


def inv_normalize(image: np.array,
                  mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)
                  ) -> np.array:
    return (image * std + mean) * 255


def tensor_to_image(tensor,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                    ) -> np.array:
    tensor = tensor.permute(1, 2, 0)
    arr = np.array(tensor)
    image = inv_normalize(arr, mean=mean, std=std)
    return image.astype(np.uint8)
