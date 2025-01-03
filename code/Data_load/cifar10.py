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
#                                 cifar10                                     #
#                                                                             #
###############################################################################


######################################### train #########################################



class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, 
                 transform=None, target_transform=None, 
                 split_percentage=0.9, random_seed=1, 
                 args=None,logger=None,num_class=10,
                 EM = False
                 ):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('../data/cifar10/train_images.npy')
        original_labels = np.load('../data/cifar10/train_labels.npy')

        
        ######## generate noisy labels ########

        if args.error_rate_type=='high' or args.error_rate_type=='mid' or args.error_rate_type=='low':

            logger.info('Cifar10: Getting instance-dependent annotations......')
            file_name = '../data/cifar10_' + str(args.R) + '_' + str(args.l) + '/' + args.error_rate_type + '_' + str(random_seed)
            annotations = np.load(file_name + '_annotations.npy')
            noisy_label = np.load(file_name + '_noisy_label.npy')
            transition_true = np.load(file_name + '_transition_true.npy')

            print('Data: ', file_name)
        
        elif args.error_rate_type=='real':

            logger.info('Cifar10: Getting real annotations......')
            annotations = np.load('../data/cifar_n/annotations_cifar10n.npy')
            annotations = annotations.astype(int)
            noisy_label = np.load('../data/cifar_n/majority_vote_cifar10n.npy')
        
        else:
            logger.info('Wrong choice')

        
        ######## split: train and validation ########

        num_samples = int(original_images.shape[0])
        rng = default_rng(random_seed)
        train_set_index = rng.choice(num_samples, int(num_samples * split_percentage), replace=False)
        index_all = np.arange(num_samples)
        val_set_index = np.delete(index_all, train_set_index)


        if self.train:

            self.train_data = original_images[train_set_index]
            self.train_labels = original_labels[train_set_index]
            if EM == True:
                EM_labels = utils.tools.DS_EM(annotations, original_labels, num_class, seed=random_seed, noisy_label=noisy_label)
                self.train_noisy_label = EM_labels[train_set_index]
            else:
                self.train_noisy_label = noisy_label[train_set_index]

            self.train_label_trusted_1 = -1 * np.ones(len(train_set_index))
            self.train_label_trusted_2 = -1 * np.ones(len(train_set_index))
            self.train_annotations = annotations[train_set_index]

            print('error rate (train):', (self.train_noisy_label != original_labels[train_set_index]).sum() / self.train_noisy_label.shape[0])

            # transition
            if args.error_rate_type!="real":
                self.train_transition_true = transition_true[train_set_index]
                print("shape of transitions (train)", self.train_transition_true.shape)

        else:

            self.val_data = original_images[val_set_index]
            self.val_labels = original_labels[val_set_index]
            if EM == True:
                EM_labels = utils.tools.DS_EM(annotations, original_labels, num_class, seed=random_seed, noisy_label=noisy_label)
                self.val_noisy_label = EM_labels[val_set_index]
            else:
                self.val_noisy_label = noisy_label[val_set_index]
            
            print('error rate (val):', (self.val_noisy_label != original_labels[val_set_index]).sum() / self.val_noisy_label.shape[0])

            # transition
            if args.error_rate_type!="real":
                self.val_transition_true = transition_true[val_set_index]
                print("shape of transitions (validation)", self.val_transition_true.shape)
                
        
    
    def __getitem__(self, index):
        
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            trusted_label_1, trusted_label_2 = self.train_label_trusted_1[index], self.train_label_trusted_2[index]
            noisy_label, annot = self.train_noisy_label[index], self.train_annotations[index]

            img = Image.fromarray(img)

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

            img = Image.fromarray(img)

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
    
        
class cifar10_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        print(" ")
        print("----------------------- Cifar10 (test) -----------------------")

        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('../data/cifar10/test_images.npy')
        self.test_labels = np.load('../data/cifar10/test_labels.npy')
        
        print("test data shape:", self.test_data.shape)


    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)



        