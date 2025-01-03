import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

import numpy as np
from numpy import genfromtxt
from numpy.matlib import repmat
from numpy.random import default_rng

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from PIL import Image
from scipy import stats
from pathlib import Path

from Data_load import transformer 
from utils import synthetic 
import utils.tools, pdb



######################################### train #########################################



class animal10N_dataset(Data.Dataset):
    def __init__(self, train=True, 
                 transform=None, target_transform=None, 
                 split_percentage=0.9, random_seed=1, 
                 args=None,logger=None,num_class=10,
                 EM = False
                 ):
        
        self.train = train
        self.target_transform = transform_y
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dir = '../data/animal/training/' 

        raw_data = np.load('../data/animal/train_data.npy')
        labels = np.load('../data/animal/train_labels.npy')

        ######## split: train and validation ########

        num_samples = int(len(raw_data))
        rng = default_rng(random_seed)
        train_set_index = rng.choice(num_samples, int(num_samples * split_percentage), replace=False)
        index_all = np.arange(num_samples)
        val_set_index = np.delete(index_all, train_set_index)

        if self.train:
            self.train_data = raw_data[train_set_index]
            self.train_labels = labels[train_set_index]
            self.train_noisy_label = labels[train_set_index]
            self.train_annotations = labels[train_set_index].reshape((len(train_set_index), 1))
            self.train_label_trusted_1 = -1 * np.ones(len(train_set_index))
            self.train_label_trusted_2 = -1 * np.ones(len(train_set_index))
            
        else:
            self.val_data = raw_data[val_set_index]
            self.val_labels = labels[val_set_index]
            self.val_noisy_label = labels[val_set_index]


    
    def __getitem__(self, index):

        if self.train:
            img_id = self.train_data[index]
            img_path = self.dir + img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)

            label = self.train_labels[index]
            trusted_label_1, trusted_label_2 = self.train_label_trusted_1[index], self.train_label_trusted_2[index]
            noisy_label, annot = self.train_noisy_label[index], self.train_annotations[index]

            label = self.target_transform(label)
            trusted_label_1 = self.target_transform(trusted_label_1)
            trusted_label_2 = self.target_transform(trusted_label_2)
            annot = self.target_transform(annot)
            noisy_label = self.target_transform(noisy_label)

            return index, img, label, trusted_label_1, trusted_label_2, annot, noisy_label
        
        else:
            img_id = self.val_data[index]
            img_path = self.dir + img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)

            label, noisy_label = self.val_labels[index], self.val_noisy_label[index]
            label = self.target_transform(label)
            noisy_label = self.target_transform(noisy_label)

            return index, img, label, noisy_label

    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)
	
    def update_trusted_label(self, trusted_label, idx, model_idx):
        trusted_label = torch.from_numpy(trusted_label).float()
        if self.train:
            if model_idx == "1":
                self.train_label_trusted_1[idx] = trusted_label
            elif model_idx == "2":
                self.train_label_trusted_2[idx] = trusted_label
        else:
            print("################## Wrong! ##################")


    def get_true_transition(self, idx):
        if self.train:
            return torch.tensor(self.train_transition_true[idx])
        else:
            return torch.tensor(self.val_transition_true[idx])
    
    def get_annot(self, idx):
        if self.train:
            return torch.tensor(self.train_annotations[idx])
        else:
            return 
    
    def get_noisy_label(self, idx):
        if self.train:
            return torch.tensor(self.train_noisy_label[idx])
        else:
            return torch.tensor(self.val_noisy_label[idx])
    
    def get_trusted_label(self, idx, model_idx):
        if self.train:
            if model_idx == "1":
                return torch.tensor(self.train_label_trusted_1[idx])
            elif model_idx == "2":
                return torch.tensor(self.train_label_trusted_2[idx])
        else:
            return 
    
        
       
    
######################################### test #########################################
    
        
class animal10N_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        print(" ")
        print("----------------------- animal10N (test) -----------------------")

        self.target_transform = transform_y
        self.transform_test = transforms.Compose([
            # transforms.Resize(64),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Normalize((0.6959, 0.6537, 0.6371),
            #                      (0.3113, 0.3192, 0.3214)),
        ])

        self.dir = '../data/animal/testing/' 
        self.test_data = np.load('../data/animal/test_data.npy')
        self.test_labels = np.load('../data/animal/test_labels.npy')
           

    def __getitem__(self, index):
        
        img_id = self.test_data[index]
        img_path = self.dir + img_id
        image = Image.open(img_path).convert('RGB')
        img = self.transform_test(image)

        label = self.test_labels[index]
        label = self.target_transform(label)

        return img, label

    
    def __len__(self):
        return len(self.test_data)




def transform_y(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target



        
