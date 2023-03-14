import nn_util
import torch
from torch import nn


class Convnet(nn.Module):
    def __init__(self, device, lr = 0.001, d = 17, num_of_classes = 10, channels = 64):
        super(Convnet, self).__init__()
        self.device = device
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

    def get_embeddings(self):
        return self.embeddings(torch.tensor(range(self.num_of_classes), device=self.device))

    def forward(self, x):
        return torch.cat((self.model(x), self.get_embeddings()), dim=0)