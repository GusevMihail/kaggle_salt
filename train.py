import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss

import models
from dataset import train_dataset, valid_dataset

# file_list_val = file_list[::10]
# file_list_train = [f for f in file_list if f not in file_list_val]
# dataset = TGSSaltDataset(train_path, file_list_train)
# dataset_val = TGSSaltDataset(train_path, file_list_val)

device = 'cpu'

model = models.unet11()
model.to(device)
#

learning_rate = 1e-4
loss_fn = BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch=1):
    model.train(True)
    for e in range(epoch):
        train_loss = []
        for image, mask in tqdm.tqdm(DataLoader(train_dataset, batch_size=4, shuffle=True)):
            image = image.type(torch.float).to(device)
            y_pred = model(image)
            loss = loss_fn(y_pred, mask.to(device))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

        val_loss = []
        for image, mask in DataLoader(valid_dataset, batch_size=50, shuffle=False):
            image = image.to(device)
            y_pred = model(image)

            loss = loss_fn(y_pred, mask.to(device))
            val_loss.append(loss.item())

        print("Epoch: %d, Train: %.3f, Val: %.3f" % (e, np.mean(train_loss), np.mean(val_loss)))
    model.train(False)
