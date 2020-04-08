import torch

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


def load_model(path, model, optimizer, scheduler, train_history=None, val_history=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if train_history:
        train_history = checkpoint['train_history']
        val_history = checkpoint['val_history']
    print('successfully loaded')