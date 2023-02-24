import torch
from torch import nn
from torch import optim
import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import math as Math

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# if (device.type == 'cuda'):
#     print('Using GPU')
# else:
#     print('Using CPU')
    
num_epochs = 5

def conv_layer(input, output, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride),
        nn.ReLU()
    )

# def set_gpu(x):
#     os.environ['CUDA_VISIBLE_DEVICES'] = x
#     print('using gpu:', x)

def conv_out_size(kernel_size, stride, padding, input_size):
    return Math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

def conv_final_out_size(conv_layers, kernel_size, stride, padding, input_size):
    if (conv_layers < 1):
        return input_size
    return conv_final_out_size(conv_layers-1, kernel_size, stride, padding, conv_out_size(kernel_size, stride, padding, input_size))

def get_final_layers_size(picture_size, previous_layer_size):
    return picture_size * picture_size * previous_layer_size

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.lr = 0.001
        self.k_size = 4
        self.stride = 1
        self.padding = 0
        self.in_size = 1
        self.pic_size = 28
        self.out_size = 17
        self.middle_layers = [64, 64, 64, 64]
        self.model = nn.Sequential(
            # 1 x 28 x 28 --> 4 x 13 x 13
            conv_layer(self.in_size, self.middle_layers[0], self.k_size, self.stride),
            conv_layer(self.middle_layers[0], self.middle_layers[1], self.k_size, self.stride),
            conv_layer(self.middle_layers[1], self.middle_layers[2], self.k_size, self.stride),
            # 4 x 13 x 13 --> 8 x 5 x 5 
            conv_layer(self.middle_layers[2], self.middle_layers[3], self.k_size, self.stride),
            # 8 x 5 x 5 --> 10 x 1 x 1 
            #conv_layer(self.middle_layers[3], self.out_size, self.k_size, self.stride),
            nn.Flatten(),
            nn.Linear(get_final_layers_size(
                conv_final_out_size(len(self.middle_layers), self.k_size, self.stride, self.padding, self.pic_size), 
                self.middle_layers[-1]), self.out_size)
        )

        self.num_of_classes = 10
        self.d = self.out_size

        self.embeddings = nn.Embedding(self.num_of_classes, self.d)

    def forward(self, x):
        x = self.model(x)
        y = self.embeddings(torch.tensor(range(self.num_of_classes), device=device))
        return torch.cat((x, y), dim=0)


# def my_custom_loss(output, target):
#     loss = torch.mean((output-target*2)**3)
#     return loss

def simple_dist_loss(output, target, num_of_classes, target_class_map):
    acc_loss = torch.tensor(0.0, requires_grad=True, device=device)
    acc_loss_div = torch.zeros(output.shape, device=device, dtype=torch.float)

    for i, output_embedding in enumerate(output[:-num_of_classes]):
        actual_index = target_class_map[target[i].item()] - num_of_classes
        actual_embedding = output[actual_index]

        diff = output_embedding - actual_embedding
        squared_dist = (diff).pow(2).sum(0)
        squared_dist_div = diff

        acc_loss_div[i] = squared_dist_div
        acc_loss_div[actual_index] = acc_loss_div[actual_index] - squared_dist_div

        acc_loss = acc_loss + squared_dist

    # return torch.tensor(acc_loss, requires_grad=True, device=device)
    return acc_loss, acc_loss_div


def main():
    if (device.type == 'cuda'):
        print('Using GPU')
    else:
        print('Using CPU')

    model = Convnet().to(device)
    optimiser = optim.Adam(model.parameters(), lr=model.lr)
    loss_func = simple_dist_loss
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

    total_step = len(loaders['train'])
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(loaders["train"]):
            images = images.to(device)
            labels = labels.to(device)

            res = model(images)
            loss, loss_div = loss_func(res, labels, model.num_of_classes, { i:i for i in range(model.num_of_classes) })
            optimiser.zero_grad()
            res.backward(gradient = loss_div)
            optimiser.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               
                pass
    

    
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

        accuracy = correct / total
        print('Test Accuracy of the model on the 10000 test images: %.3f%' % accuracy * 100)    

    return

if __name__ == '__main__':
    main()