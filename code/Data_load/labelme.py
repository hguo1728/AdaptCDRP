import torch
import torch.utils.data as Data

import numpy as np
from numpy import genfromtxt
from numpy.matlib import repmat
from numpy.random import default_rng

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from PIL import Image
from scipy import stats
from pathlib import Path

# from Data_load import transformer 
from utils import synthetic 
import utils.tools, pdb


###############################################################################
#                                                                             #
#                                 LabelMe                                     #
#                                                                             #
###############################################################################


######################################### train #########################################

class labelme_dataset(Data.Dataset):
    def __init__(self, train=True, 
                 transform=None, target_transform=None, 
                 split_percentage=0.9, random_seed=1, 
                 args=None,logger=None,num_class=8,
                 EM = False
                 ):
        
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        
        if self.train:

            print(" ")
            print("----------------------- LabelMe (train) -----------------------")

            self.train_data = np.load('../data/labelme/prepared/data_train_vgg16.npy')
            self.train_labels = np.load('../data/labelme/prepared/labels_train.npy')

            # annotation
            logger.info('LabelMe: Getting real annotations......')
            annotations = np.load('../data/labelme/prepared/answers.npy')
            self.train_annotations = annotations.astype(int)
            
            # noisy label: majority vote or EM
            if EM:
                EM_labels = utils.tools.DS_EM(annotations, self.train_labels, num_class, seed=random_seed, noisy_label=noisy_label)
                self.train_noisy_label = EM_labels
            else:
                noisy_label = np.load('../data/labelme/prepared/labels_train_mv.npy')
                self.train_noisy_label = noisy_label
            
            self.train_label_trusted_1 = -1 * np.ones(len(self.train_labels))
            self.train_label_trusted_2 = -1 * np.ones(len(self.train_labels))

            print('error rate (train):', (self.train_noisy_label != self.train_labels).sum() / self.train_noisy_label.shape[0])
            print("shape of annotations (train)", self.train_annotations.shape)
        
        else:
    
            print(" ")
            print("----------------------- LabelMe (validation) -----------------------")

            self.val_data = np.load('../data/labelme/prepared/data_valid_vgg16.npy')
            self.val_labels= np.load('../data/labelme/prepared/labels_valid.npy')
            self.val_noisy_label= np.load('../data/labelme/prepared/labels_valid.npy')



    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            trusted_label_1, trusted_label_2 = self.train_label_trusted_1[index], self.train_label_trusted_2[index]
            noisy_label, annot = self.train_noisy_label[index], self.train_annotations[index]

            # img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                label = self.target_transform(label)
                trusted_label_1 = self.target_transform(trusted_label_1)
                trusted_label_2 = self.target_transform(trusted_label_2)
                annot = self.target_transform(annot)
                noisy_label = self.target_transform(noisy_label)
            
            return index, img, label, trusted_label_1, trusted_label_2, annot, noisy_label
                            
        else:
            img, label, noisy_label = self.val_data[index], self.val_labels[index], self.val_noisy_label[index]

            # img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
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

        
class labelme_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None, test=True, return_idx = False):

        self.transform = transform
        self.target_transform = target_transform
        # self.test = test
        # self.return_idx = return_idx

        print(" ")
        print("----------------------- LabelMe (test) -----------------------")
        
        self.test_data = np.load('../data/labelme/prepared/data_test_vgg16.npy')
        self.test_labels= np.load('../data/labelme/prepared/labels_test.npy')

        print("test data shape:", self.test_data.shape)



    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
    
        return img, label
    
    
    def __len__(self):
        return len(self.test_data)
       

        
