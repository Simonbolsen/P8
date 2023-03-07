import nn_util
import torch
from torch import nn


class Convnet(nn.Module):
    def __init__(self, device, lr, embedding_out_size, num_of_classes, 
                 out_channels, kernel_size, stride,
                 in_dimensions, img_size, linear_layers,
                 linear_layers_size, padding = 0):
        super(Convnet, self).__init__()
        self.device = device
        
        self.lr = lr
        self.d = embedding_out_size
        
        self.num_of_classes = num_of_classes

        self.channels = out_channels
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.in_size = in_dimensions
        self.pic_size = img_size
        
        self.middle_layers = [self.channels, self.channels, self.channels, self.channels]
        self.final_conv_out_size = nn_util.conv_final_out_size(len(self.middle_layers), self.k_size, self.stride, self.padding, self.pic_size)
        
        self.linear_layers = linear_layers
        self.linear_first_in = nn_util.get_final_layers_size(
                                self.final_conv_out_size,
                                self.middle_layers[-1]
                                )
         
        self.linear_layers_size = linear_layers_size
        
        self.model = nn.Sequential(
            # 1 x 28 x 28
            nn_util.conv_layer(self.in_size, self.middle_layers[0], self.k_size, self.stride),
            nn_util.conv_layer(self.middle_layers[0], self.middle_layers[1], self.k_size, self.stride),
            nn_util.conv_layer(self.middle_layers[1], self.middle_layers[2], self.k_size, self.stride),
            nn_util.conv_layer(self.middle_layers[2], self.middle_layers[3], self.k_size, self.stride),
            nn.Flatten(),
            nn_util.create_n_linear_layers(self.linear_layers, self.linear_first_in, self.linear_layers_size),
            nn.Linear(self.linear_layers_size, self.d)
        )

        self.embeddings = nn.Embedding(self.num_of_classes, self.d)

    def forward(self, x):
        x = self.model(x)
        y = self.embeddings(torch.tensor(range(self.num_of_classes), device=self.device))
        return torch.cat((x, y), dim=0)