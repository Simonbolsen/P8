
import time
from torch import device
import torch


def train_model(model, dataloaders, loss_func, optimizer, num_epochs):
    since = time.time()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(torch.cuda.is_available())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        

        #Set the mode
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
                
                    _, preds = torch.max(outputs, 1)       

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()         

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


        
