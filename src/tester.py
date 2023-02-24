import torch
from torch import nn
from torch import optim
import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import nn_util

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Convnet(nn.Module):
    def __init__(self, lr = 0.001, d = 17, num_of_classes = 10, channels = 64):
        super(Convnet, self).__init__()
        self.lr = lr
        self.d = d
        self.num_of_classes = num_of_classes
        self.channels = channels
        self.k_size = 4
        self.stride = 1
        self.padding = 0
        self.in_size = 1
        self.pic_size = 28
        self.middle_layers = [self.channels, self.channels, self.channels, self.channels]
        self.model = nn.Sequential(
            # 1 x 28 x 28 --> 4 x 13 x 13
            nn_util.conv_layer(self.in_size, self.middle_layers[0], self.k_size, self.stride),
            nn_util.conv_layer(self.middle_layers[0], self.middle_layers[1], self.k_size, self.stride),
            nn_util.conv_layer(self.middle_layers[1], self.middle_layers[2], self.k_size, self.stride),
            # 4 x 13 x 13 --> 8 x 5 x 5 
           nn_util. conv_layer(self.middle_layers[2], self.middle_layers[3], self.k_size, self.stride),
            # 8 x 5 x 5 --> 10 x 1 x 1 
            #conv_layer(self.middle_layers[3], self.d, self.k_size, self.stride),
            nn.Flatten(),
            nn.Linear(nn_util.get_final_layers_size(
                nn_util.conv_final_out_size(len(self.middle_layers), self.k_size, self.stride, self.padding, self.pic_size), 
                self.middle_layers[-1]), self.d)
        )

        self.embeddings = nn.Embedding(self.num_of_classes, self.d)

    def forward(self, x):
        x = self.model(x)
        y = self.embeddings(torch.tensor(range(self.num_of_classes), device=device))
        return torch.cat((x, y), dim=0)

def main():
    if (device.type == 'cuda'):
        print('Using GPU')
    else:
        print('Using CPU')

    num_epochs = 1
    model = Convnet().to(device)
    optimiser = optim.Adam(model.parameters(), lr=model.lr)
    loss_func = nn_util.simple_dist_loss
    target_class_map = { i:i for i in range(model.num_of_classes) }

    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

    loaders = {
        "train": torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        "test": torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
    }

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    train(model, num_epochs, loaders, optimiser, loss_func)
    
    accuracy = eval(model, loaders, target_class_map)
    print(f'Test Accuracy of the model on the 10000 test images: {(accuracy * 100):.2f}%')    
    return

def train(model, num_epochs, loaders, optimiser, loss_func): 
    total_step = len(loaders['train'])
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(loaders["train"]):
            images = images.to(device)
            labels = labels.to(device)

            res = model(images)
            loss, loss_div = loss_func(res, labels, model.num_of_classes, { i:i for i in range(model.num_of_classes) }, device)
            optimiser.zero_grad()
            res.backward(gradient = loss_div)
            optimiser.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               
                pass

def eval(model, loaders, target_class_map):
     # Test the model
    model.eval()    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            images = images.to(device)
            labels = labels.to(device)

            test_output = model(images)

            for i, output_embedding in enumerate(test_output[:-model.num_of_classes]):
                smallest_sqr_dist = 100000000
                smallest_k = 0
                for k in range(model.num_of_classes):
                    actual_class_embedding = test_output[k - model.num_of_classes]
                    squared_dist = (actual_class_embedding - output_embedding).pow(2).sum(0)
                    
                    if squared_dist < smallest_sqr_dist:
                        smallest_sqr_dist = squared_dist
                        smallest_k = k

                if smallest_k == target_class_map[labels[i].item()]:
                    correct += 1
                total += 1
        return correct / total

if __name__ == '__main__':
    main()