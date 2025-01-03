import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.hub as hub
import torchvision.models as models
from torch.autograd import Variable
import math
from torchvision import transforms
from torch.nn.utils import prune


class Transition(nn.Module):
    
    def __init__(self, dim_in, dim_hidden, num_classes, R, FC_2=True):
        """
        mnist:
            dim_in: 1176 or 400
            dim_hidden: 512 or 256

        cifar10/100:
            dim_in: 2048
            dim_hidden: 512 or 1024
        """
        super(Transition,self).__init__()

        self.num_classes = num_classes
        self.R = R

        if FC_2:

            self.fc1_1 = nn.Linear(dim_in, dim_hidden)
            self.fc2_1 = nn.Linear(dim_hidden, dim_hidden // 2)

            self.fc1_2 = nn.Linear(dim_in, dim_hidden)
            self.fc2_2 = nn.Linear(dim_hidden, dim_hidden // 2)

            self.relu = nn.LeakyReLU(0.2)
            self.drop_out = nn.Dropout(0.5)

            # self.c1 = nn.Sequential(
            #     nn.Linear(dim_in, dim_hidden),
            #     nn.Dropout(0.5),
            #     nn.LeakyReLU(0.2), # nn.ReLU(),

            #     nn.Linear(dim_hidden, dim_hidden // 2),
            #     nn.Dropout(0.5),
            #     nn.LeakyReLU(0.2), # nn.ReLU(),
            # )
            # self.c2 = nn.Sequential(
            #     nn.Linear(dim_in, dim_hidden),
            #     nn.Dropout(0.5),
            #     nn.LeakyReLU(0.2), # nn.ReLU(),

            #     nn.Linear(dim_hidden, dim_hidden // 2),
            #     nn.Dropout(0.5),
            #     nn.LeakyReLU(0.2), # nn.ReLU(),
            # )
        
        else:
            self.c1 = nn.Sequential(
                nn.Linear(dim_in, dim_hidden),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2), # nn.ReLU()
            )
            self.c2 = nn.Sequential(
                nn.Linear(dim_in, dim_hidden // 2),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2), # nn.ReLU()
            )

        self.out_1 = nn.Linear(dim_hidden // 2, num_classes * num_classes, bias=False)
        self.out_2 = nn.Linear(dim_hidden // 2, num_classes * R, bias=False)

        self.prune_flag = 0
        self.initialize_weights()

    def forward(self, x):
        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                para.data[self.mask[name]] = 0
        

        out1 = self.relu(self.fc1_1(x))
        out1 = self.drop_out(out1)
        out1 = self.relu(self.fc2_1(out1))
        out1 = self.drop_out(out1)
        out1 = self.out_1(out1)

        out2 = self.relu(self.fc1_2(x))
        out2 = self.drop_out(out2)
        out2 = self.relu(self.fc2_2(out2))
        out2 = self.drop_out(out2)
        out2 = self.out_1(out2)

        # out1 = self.c1(x)
        # out1 = self.fc1(out1)
        # out1 = out1.reshape(out1.size(0), 1, self.num_classes, self.num_classes)

        # out2 = self.c2(x)
        # out2 = self.fc2(out2)
        # out2 = out2.reshape(out2.size(0), self.R, 1, self.num_classes)

        out1 = out1.reshape(out1.size(0), self.K, self.K) # (n, K, K)
        out2 = out2.reshape(out2.size(0), self.R, self.K) # (n, R, K)

        out = out1[:, None, :, :] + out2[:, :, None, :]
        prob = F.softmax(out, dim=3) # (n, R, K, K)

        return prob
    

    
    def initialize_weights(self):
        for m in self.modules():
        
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_prune(self):
        self.prune_flag = 1

    def cancel_prune(self):
        self.prune_flag = 0


def transition_posterior_prune(model, threshold):
    
    # mask == 0: cut

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and (name[:4] != 'out_'):

            mask_weight = torch.abs(module.state_dict()["weight"]) > threshold  # > threshold: keep the param
            prune.custom_from_mask(module=module, name='weight', mask=mask_weight)

            mask_bias = torch.abs(module.state_dict()["bias"]) > threshold  # > threshold: keep the param
            prune.custom_from_mask(module=module, name='bias', mask=mask_bias)


    # num_linear_layers_1 = len(model.c1) // 3
    # layer_idx = 0

    # for i in range(num_linear_layers_1):
    #     mask_weight = torch.abs(model.c1[layer_idx].weight) > threshold
    #     prune.custom_from_mask(module=model.c1[layer_idx], name='weight', mask=mask_weight)

    #     mask_bias = torch.abs(model.c1[layer_idx].bias) > threshold
    #     prune.custom_from_mask(module=model.c1[layer_idx], name='bias', mask=mask_bias)
        
    #     layer_idx += 3
    
    # num_linear_layers_2 = len(model.c2) // 3
    # layer_idx = 0
    
    # for i in range(num_linear_layers_2):
    #     mask_weight = torch.abs(model.c2[layer_idx].weight) > threshold
    #     prune.custom_from_mask(module=model.c2[layer_idx], name='weight', mask=mask_weight)

    #     mask_bias = torch.abs(model.c2[layer_idx].bias) > threshold
    #     prune.custom_from_mask(module=model.c2[layer_idx], name='bias', mask=mask_bias)
        
    #     layer_idx += 3


# class FCNN(nn.Module):
    
#     def __init__(self, R, K, hidden_units, hidden_layers):
#         super(FCNN,self).__init__()
#         layer_list=[]
#         n_in = R
#         for i in range(hidden_layers):
#             layer_list.append(nn.Linear(n_in, hidden_units))
#             layer_list.append(nn.ReLU(inplace=False))
#             n_in = hidden_units
#         layer_list.append(nn.Linear(hidden_units,K))
#         self.layers=nn.Sequential(*layer_list)
        
#     def forward(self, x):
#         x = self.layers(x)
#         prob = F.softmax(x, dim=1)
#         return prob

