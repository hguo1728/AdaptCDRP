import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from torch.autograd import Variable

import argparse
import logging
# from tqdm import tqdm

import numpy as np
from numpy.random import default_rng
import random
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

from utils.synthetic import *
from utils.tools import *
from Data_load.cifar10 import *
from Data_load.cifar100 import *
from Data_load.labelme import *
from Data_load.transformer import *
from models import LeNet, ResNet, VGG, Transition, FCNN


################################################################################################
#                         instance-independent: frequency counting                             #
################################################################################################


def get_transition_a(args, alg_options, logger):

    N = alg_options['train_dataset'].__len__()
    K = args.num_classes
    device = alg_options['device']

    # annot -> one hot annot 

    if args.one_transition == "true":
        R = 1
        annot_train = torch.tensor(alg_options['train_dataset'].train_noisy_label).to(device) # (N, )
        annot_one_hot = F.one_hot(annot_train.long(), K).float() # (N, K)
        annot_one_hot = annot_one_hot.unsqueeze(1) # (N, 1, K)

    else:
        R = args.R
        annot_train = alg_options['train_dataset'].get_annot(idx=range(N)) # (N, R)
        annot_one_hot = torch.zeros((N * R, K)).to(device)
        annot_train = annot_train.reshape(-1).to(device)
        mask = (annot_train != -1)
        annot_one_hot[mask] = F.one_hot(annot_train[mask].long(), K).float()
        annot_one_hot = annot_one_hot.reshape(N, R, K)

    # trusted label -> one hot trusted

    # model 1: train
    trusted_labels_1 = alg_options['train_dataset'].get_trusted_label(idx=range(N), model_idx="1").to(device) # (N, )
    trusted_one_hot_1 = torch.zeros((N, K)).to(device) # (n, K)
    mask = (trusted_labels_1 != -1)
    trusted_one_hot_1[mask] = F.one_hot(trusted_labels_1[mask].long(), K).float()

    # model 2: train
    trusted_labels_2 = alg_options['train_dataset'].get_trusted_label(idx=range(N), model_idx="2").to(device) # (N, )
    trusted_one_hot_2 = torch.zeros((N, K)).to(device) # (n, K)
    mask = (trusted_labels_2 != -1)
    trusted_one_hot_2[mask] = F.one_hot(trusted_labels_2[mask].long(), K).float()

    # error rates: (pi^{(r)}_{j, l}) -- the r-th annotatot; true label j; noisy annotation l
    transition_1 = torch.zeros((R, K, K))
    transition_2 = torch.zeros((R, K, K))

    for r in range(R): # annotator: r
        for j in range(K): # true label: j
            for l in range(K): # noisy label: l
                transition_1[r, j, l] = torch.dot(trusted_one_hot_1[:, j], annot_one_hot[:, r, l])
                transition_2[r, j, l] = torch.dot(trusted_one_hot_2[:, j], annot_one_hot[:, r, l])
            
            # normalize: summing over all obervation classes
            sum_temp_1 = torch.sum(transition_1[r, j, :])
            sum_temp_2 = torch.sum(transition_2[r, j, :])
            if sum_temp_1 > 0:
                transition_1[r, j, :] /= float(sum_temp_1)
            if sum_temp_2 > 0:
                transition_2[r, j, :] /= float(sum_temp_2)

    transition = ((transition_1 + transition_2) / 2)

    if args.dataset != "animal":

        logger.info("*---*---*---*---* estimation error (transition) *---*---*---*---*")

        if args.one_transition == "true": # (1, K, K)

            # calculate T_true
            T_true = torch.zeros((1, K, K)).to(device)

            noisy_one_hot = annot_one_hot.squeeze(1) # (N, 1, K) -> (N, K)
            labels = torch.tensor(alg_options['train_dataset'].train_labels[range(N)]).to(device) # (N, )
            labels_one_hot = F.one_hot(labels.long(), K).float() # (N, K)

            for j in range(K): # true label: j
                for l in range(K): # noisy label: l
                    T_true[0, j, l] = torch.dot(labels_one_hot[:, j], noisy_one_hot[:, l])
                
                sum_temp_1 = torch.sum(T_true[0, j, :])
                if sum_temp_1 > 0:
                    T_true[0, j, :] /= float(sum_temp_1)
            
            diff_abs = torch.abs(transition.to(device) - T_true.to(device)).squeeze(0) # (1, K, K) -> (K, K)
            logger.info('One Transition -- estimation error (train):{}'.format(diff_abs))

            estimation_error = diff_abs.max(1)[0].max(0)[0]
            logger.info('estimation error (train):{}'.format(estimation_error))
            
        else: # (R, K, K)

            if args.error_rate_type == 'real':
                if args.dataset == "cifar10":
                    T_true = torch.load("../data/cifar_n/cifar10_T.npy") # (3, K, K)
                elif args.dataset == "cifar100":
                    T_true = torch.load("../data/cifar_n/cifar100_T.npy") # (1, K, K)
                elif args.dataset == "labelme":
                    T_true = torch.load("../data/labelme/prepared/labelme_T.npy") # (59, K, K)
                
                diff_abs = torch.abs(transition.to(device) - T_true.to(device))
                estimation_error = diff_abs.max(2)[0].max(1)[0]
                logger.info('estimation error (train):{}'.format(estimation_error))

            else:
                train_num = 0
                total_estimation_error = torch.zeros(args.R)
                train_loader = alg_options['train_loader']

                for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

                    train_num += batch_y.shape[0]

                    batch_T_true = alg_options['train_dataset'].get_true_transition(indexes).to(device) # (n, R, K)
                    batch_T_est =  transition[:, batch_y, :].detach() # (R, n, K)
                    batch_T_est = batch_T_est.permute((1, 0, 2)) # (n, R, K)
                    batch_estimation_error = torch.max(torch.abs(batch_T_true - batch_T_est.to(device)), 2)[0].sum(0) # total estimation error on the batch
                    total_estimation_error += batch_estimation_error.cpu()
                
                estimation_error = (total_estimation_error).detach().numpy() / float(train_num)
                logger.info('estimation error (train):{}'.format(estimation_error))
    else:
        estimation_error = 0
        print("Animal: no estimation error")

    return transition, estimation_error






################################################################################################
#                instance-independent: BayesianIDNT (Guo-Yi-Wang 2023)                         #
################################################################################################


def get_transition_b(args, alg_options, logger, t, model):

    N = alg_options['train_dataset'].__len__()
    K = args.num_classes
    device = alg_options['device']
    train_loader = alg_options['train_loader']

    if args.one_transition == "true":
        R = 1
    else:
        R = args.R

    if args.dataset == "cifar10":
        model_T = ResNet.ResNet18_T(num_classes=10, R=R)

    elif args.dataset == "cifar100":
        model_T = ResNet.ResNet34_T(num_classes=100, R=R)

    elif args.dataset == "labelme":
        model_T = FCNN.FCNN_T(K=8, R=R)

    else:
        logger.info('Incorrect choice for dataset')
    
    model_T = model_T.to(device)
    optimizer_T = torch.optim.Adam(model_T.parameters(), lr=args.learning_rate_T)
    scheduler_T = optim.lr_scheduler.OneCycleLR(optimizer_T, args.learning_rate_T, epochs=args.n_epoch_T_init+args.n_epoch_T_fine_tune, steps_per_epoch=len(train_loader), verbose=True)

    logger.info("*---*---*---*---* transition matrix: train *---*---*---*---*")

    model_T.get_pretrained_weights(model)

    for epoch in range(args.n_epoch_T_init):

        # MAP......
        logger.info("MAP Training......")
        model_T.train()
        train_loss, _, _ = train_T_MAP(model_T, optimizer_T, scheduler_T, alg_options, args, logger, epoch, one_transition=args.one_transition)
        logger.info("Epoch [{}], Train loss (MAP): {:.4f}".format(epoch+1, train_loss))
    
    logger.info("*---*---*---*---* transition matrix: fine tune *---*---*---*---*")

    lambda_n = args.lambdan
    sigma1= args.sigma1
    sigma0= args.sigma0

    model_T.eval()
    sparse_threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma1 / sigma0)) / (0.5 / sigma0 - 0.5 / sigma1))
    posterior_prune(model_T, sparse_threshold)

    # fine tune

    for epoch in range(args.n_epoch_T_fine_tune):

        # MAP......
        logger.info("MAP Training (fine tune)......")
        model_T.train()
        loss_fine_tune, train_estimation_error, _ = train_T_MAP(model_T, optimizer_T, scheduler_T, alg_options, args, logger, epoch, masked=True, one_transition=args.one_transition)
        logger.info("Epoch [{}], Train loss (MAP, fine tune): {:.4f}".format(epoch+1, loss_fine_tune))
    
    torch.save(model_T.state_dict(), args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_transition_model.pth')

    # sparsity: 
    total_num_para = 0
    non_zero_element = 0
    for _, param in model_T.named_parameters():
        total_num_para += param.numel()
        non_zero_element += (param.abs() > 1e-6).sum()
    sparsity_rate = non_zero_element.item() / total_num_para
    print('sparsity:', sparsity_rate)

    return model_T, train_estimation_error



# train the instance-dependent transition model

def train_T_MAP(model_T, optimizer, scheduler, alg_options, args, logger, epoch, masked=False, one_transition="true"):

    device = alg_options['device']
    train_loader = alg_options['train_loader']
    val_loader = alg_options['val_loader']

    K = args.num_classes
    # R = args.R
    lambda_n = 5e-9
    sigma1 = 5e-1    
    sigma0 = 2e-7
    c1 = (lambda_n / (1 - lambda_n)) * np.sqrt(sigma0 / sigma1)
    c2 = 0.5 * (1 / sigma0 - 1 / sigma1)

    total_batch = len(train_loader)

    train_total_loss = 0
    train_num = 0

    loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean')

    for batch_idx, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

        model_T.train()

        if torch.cuda.is_available:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device) # (n,)
            batch_trusted_label_1 = trusted_label_1.to(device) # (n,)
            batch_trusted_label_2 = trusted_label_2.to(device) # (n,)
            batch_annot = annot.to(device) # (n, R)
            noisy_label = noisy_label.to(device)
        
        
        mask_1 = (batch_trusted_label_1 != -1)
        mask_2 = (batch_trusted_label_2 != -1)

        batch_T = model_T(batch_x) # (n, R, K, K)
        batch_T_1 = batch_T[mask_1] 
        batch_T_2 = batch_T[mask_2] 
        batch_T_cat = torch.cat((batch_T_1, batch_T_2), 0)

        if args.one_transition == "true":
            batch_annot_1 = noisy_label[mask_1] # (n, )
            batch_annot_2 = noisy_label[mask_2]
            batch_annot_cat = torch.cat((batch_annot_1, batch_annot_2), 0).unsqueeze(1) # (n, ) => (n, 1)
        else:
            batch_annot_1 = batch_annot[mask_1] # (n, R)
            batch_annot_2 = batch_annot[mask_2]
            batch_annot_cat = torch.cat((batch_annot_1, batch_annot_2), 0) 

        trusted_label_one_hot_1 = F.one_hot(batch_trusted_label_1[mask_1], K).float() # (|masked|, K)
        trusted_label_one_hot_2 = F.one_hot(batch_trusted_label_2[mask_2], K).float() # (|masked|, K)
        trusted_label_one_hot_cat = torch.cat((trusted_label_one_hot_1, trusted_label_one_hot_2), 0)

        batch_size = batch_T_cat.shape[0]
        train_num += batch_size

        if batch_size == 0:
            print("batch_size = 0; continue")
            continue
        

        # ----------- calculate cross entropy loss -----------

        # condition on trusted label
        batch_T_cond = torch.einsum('nrkl, nk->nrl', [batch_T_cat, trusted_label_one_hot_cat]) # (|masked|, R, K, K) * (|masked|, K) -> (|masked|, R, K)
        
        # calculate cross entropy loss
        loss = loss_fn(torch.log(batch_T_cond+1e-10).view(-1, K), batch_annot_cat.view(-1))

        batch_loss = loss.item() # average loss on the batch
        train_total_loss += batch_loss * batch_size # total loss on the batch

        optimizer.zero_grad()
        loss.backward()

        if masked: # mask has been implemented: gamma=1; use sigma1 in the prior loss 
            for name, param in model_T.named_parameters():
                if param.requires_grad == False:
                    continue
                else:
                    param.grad = param.grad.add(param, alpha=1/(sigma1 * batch_size))
        
        elif epoch>5: # mask has NOT been implemented: use posterior prob for gamma
            for name, param in model_T.named_parameters():
                if param.requires_grad == False:
                    continue
                elif name[:4] == "out_":
                    const = 1/(sigma1 * batch_size)
                else:
                    with torch.no_grad():
                        p0 = 1 / (c1 * torch.exp(param.pow(2) * c2) + 1) # P(gamma=0 | beta)
                    const = (p0.div(sigma0) + (1 - p0).div(sigma1)).div(batch_size)

                param.grad = param.grad.add(param * const)
        
        optimizer.step()
        optimizer.zero_grad()

        # Print
        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch [{}], Iter [{}/{}], Loss: {}'.format(epoch + 1, batch_idx + 1, total_batch, batch_loss))
    
    scheduler.step()
    train_loss = float(train_total_loss) / float(train_num)


    # ----------- estimation error -----------

    if args.dataset != "animal":
        
        logger.info("*---*---*---*---* estimation error (transition) *---*---*---*---*")

        if args.error_rate_type == 'real':
            if args.dataset == "cifar10":
                T_true = torch.load("../data/cifar_n/cifar10_T.npy") # (R, K, K)
            elif args.dataset == "cifar100":
                T_true = torch.load("../data/cifar_n/cifar100_T.npy")
            elif args.dataset == "labelme":
                T_true = torch.load("../data/labelme/prepared/labelme_T.npy")
            T_true = T_true.to(device)
        

        train_num = 0
        total_estimation_error = torch.zeros(args.R)
        train_loader = alg_options['train_loader']

        for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

            with torch.no_grad():

                model_T.eval()

                if torch.cuda.is_available:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device) # (batch_size,)
                batch_size = batch_x.shape[0]
                
                if args.one_transition == "true":
                    batch_T = model_T(batch_x).expand(batch_size, args.R, K, K) # (n, 1, K, K) => (n, R, K, K)
                else:
                    batch_T = model_T(batch_x) # (n, R, K, K)

                train_num += batch_size

                # condition on true label
                label_one_hot = F.one_hot(batch_y, K).float() # (n, K)
                batch_T_cond_true = torch.einsum('nrkl, nk->nrl', [batch_T, label_one_hot]) # (n, R, K, K), (n, K) -> (n, R, K)

                if args.error_rate_type == 'real':
                    temp = T_true.expand(batch_x.shape[0], args.R, K, K) # (n, R, K, K)
                    batch_T_true = torch.einsum('nrkl, nk->nrl', [temp, label_one_hot]) # (n, R, K, K), (n, K) -> (n, R, K)
                else:
                    batch_T_true = alg_options['train_dataset'].get_true_transition(indexes).to(device) # (batch_size, R, K)

                batch_estimation_error = torch.max(torch.abs(batch_T_true - batch_T_cond_true), 2)[0].sum(0) # total estimation error on the batch
                # batch_estimation_error = (torch.sum(torch.abs(batch_T_true - batch_T_cond_true), 2) / K).sum(0) # total estimation error on the batch
                total_estimation_error += batch_estimation_error.cpu()
        
        train_estimation_error = (total_estimation_error).detach().numpy() / float(train_num)
        logger.info('estimation error (train):{}'.format(train_estimation_error))


        val_num = 0
        total_estimation_error = torch.zeros(args.R)

        for indexes, batch_x, batch_y, noisy_label in val_loader:

            with torch.no_grad():

                model_T.eval()

                if torch.cuda.is_available:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device) # (batch_size,)
                batch_size = batch_x.shape[0]
                
                if args.one_transition == "true":
                    batch_T = model_T(batch_x).expand(batch_size, args.R, K, K) # (n, 1, K, K) => (n, R, K, K)
                else:
                    batch_T = model_T(batch_x) # (n, R, K, K)

                val_num += batch_size

                # condition on true label
                label_one_hot = F.one_hot(batch_y, K).float() # (n, K)
                batch_T_cond_true = torch.einsum('nrkl, nk->nrl', [batch_T, label_one_hot]) # (n, R, K, K), (n, K) -> (n, R, K)

                if args.error_rate_type == 'real':
                    temp = T_true.expand(batch_x.shape[0], args.R, K, K) # (n, R, K, K)
                    batch_T_true = torch.einsum('nrkl, nk->nrl', [temp, label_one_hot]) # (n, R, K, K), (n, K) -> (n, R, K)
                else:
                    batch_T_true = alg_options['val_dataset'].get_true_transition(indexes).to(device) # (batch_size, R, K)

                batch_estimation_error = torch.max(torch.abs(batch_T_true - batch_T_cond_true), 2)[0].sum(0) # total estimation error on the batch
                # batch_estimation_error = (torch.sum(torch.abs(batch_T_true - batch_T_cond_true), 2) / K).sum(0) # total estimation error on the batch
                total_estimation_error += batch_estimation_error.cpu()
        
        val_estimation_error = (total_estimation_error).detach().numpy() / float(val_num)
        logger.info('estimation error (val):{}'.format(val_estimation_error))
    
    else:
        val_estimation_error = 0
        train_estimation_error = 0
        print("Animal: no estimation error")
    
    
    return train_loss, train_estimation_error, val_estimation_error



# -------------------------------- get high confidence labels --------------------------------

def update_trusted_loss(model_1, model_2, alg_options, logger, ratio, device):

    train_loader = alg_options['train_loader']
    N = alg_options['train_dataset'].__len__()
    
    model_1.eval()
    model_2.eval()

    trusted_idx_list_1 = []
    trusted_labels_list_1 = []

    trusted_idx_list_2 = []
    trusted_labels_list_2 = []

    loss_fn = nn.NLLLoss(ignore_index=-1, reduction='none') 

    with torch.no_grad():

        for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
                noisy_label = noisy_label.to(device)
            
            # model 1: trusted_label_1
            probs_1 = model_1(batch_x).softmax(-1) # (n, K)

            loss_1 = loss_fn(torch.log(probs_1 + 1e-10), noisy_label)
            ind_1_sorted = torch.argsort(loss_1.data).cpu()
            trusted_idx_list_1.extend(indexes[ind_1_sorted[:int(ratio * len(loss_1))]].cpu())
            trusted_labels_list_1.extend(noisy_label.cpu()[ind_1_sorted[:int(ratio * len(loss_1))].cpu()])

            # model 2: trusted_label_2
            probs_2 = model_2(batch_x).softmax(-1) # (batch_size, K)

            loss_2 = loss_fn(torch.log(probs_2 + 1e-10), noisy_label)
            ind_2_sorted = torch.argsort(loss_2.data).cpu()
            trusted_idx_list_2.extend(indexes[ind_2_sorted[:int(ratio * len(loss_2))]].cpu())
            trusted_labels_list_2.extend(noisy_label.cpu()[ind_2_sorted[:int(ratio * len(loss_2))].cpu()])

    trusted_idx_1 = np.array(trusted_idx_list_1).astype(int)
    trusted_idx_2 = np.array(trusted_idx_list_2).astype(int)

    trusted_labels_1 = np.array(trusted_labels_list_1)
    trusted_labels_2 = np.array(trusted_labels_list_2)

    # model 1
    trusted_labels_true_1 = alg_options['train_dataset'].train_labels[trusted_idx_1]
    Train_data_num_1 = len(trusted_idx_1)
    Train_data_ACC_1 = (np.array(trusted_labels_1) == np.array(trusted_labels_true_1)).sum() / Train_data_num_1

    logger.info("Model 1 (train) -- Selected trusted examples ACC: {} (Num: {})".format(Train_data_ACC_1, Train_data_num_1))
    alg_options['train_dataset'].update_trusted_label(-1 * np.ones(N), range(N), model_idx="1")
    alg_options['train_dataset'].update_trusted_label(trusted_labels_1, trusted_idx_1, model_idx="1")


    # model 2
    trusted_labels_true_2 = alg_options['train_dataset'].train_labels[trusted_idx_2]
    Train_data_num_2 = len(trusted_idx_2)
    Train_data_ACC_2 = (np.array(trusted_labels_2) == np.array(trusted_labels_true_2)).sum() / Train_data_num_2

    logger.info("Model 2 (train) -- Selected trusted examples ACC: {} (Num: {})".format(Train_data_ACC_2, Train_data_num_2))
    alg_options['train_dataset'].update_trusted_label(-1 * np.ones(N), range(N), model_idx="2")
    alg_options['train_dataset'].update_trusted_label(trusted_labels_2, trusted_idx_2, model_idx="2")

    return 


def update_trusted_acc(model_1, model_2, alg_options, logger, threshold, device):

    train_loader = alg_options['train_loader']
    N = alg_options['train_dataset'].__len__()
    
    model_1.eval()
    model_2.eval()

    trusted_idx_list_1 = []
    trusted_labels_list_1 = []

    trusted_idx_list_2 = []
    trusted_labels_list_2 = []

    loss_fn = nn.NLLLoss(ignore_index=-1, reduction='none') 

    with torch.no_grad():

        for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
                noisy_label = noisy_label.to(device)
            
            # model 1: trusted_label_1
            probs_1 = model_1(batch_x).softmax(-1) # (n, K)
            prob_max_1 = torch.max(probs_1, dim=1) # out: (max, max_indices) <-> (max_probs, labels)
            selected_mask_1 = prob_max_1[0] > float(threshold) 
            trusted_idx_list_1.extend(indexes[selected_mask_1.cpu()])
            trusted_labels_list_1.extend(prob_max_1[1].cpu()[selected_mask_1.cpu()])


            # model 2: trusted_label_2
            probs_2 = model_2(batch_x).softmax(-1) # (batch_size, K)

            prob_max_2 = torch.max(probs_2, dim=1) # out: (max, max_indices) <-> (max_probs, labels)
            
            selected_mask_2 = prob_max_2[0] > float(threshold) 
            trusted_idx_list_2.extend(indexes[selected_mask_2.cpu()])
            trusted_labels_list_2.extend(prob_max_2[1].cpu()[selected_mask_2.cpu()])

    trusted_idx_1 = np.array(trusted_idx_list_1).astype(int)
    trusted_idx_2 = np.array(trusted_idx_list_2).astype(int)

    trusted_labels_1 = np.array(trusted_labels_list_1)
    trusted_labels_2 = np.array(trusted_labels_list_2)

    # model 1
    trusted_labels_true_1 = alg_options['train_dataset'].train_labels[trusted_idx_1]
    Train_data_num_1 = len(trusted_idx_1)
    Train_data_ACC_1 = (np.array(trusted_labels_1) == np.array(trusted_labels_true_1)).sum() / Train_data_num_1

    logger.info("Model 1 (train) -- Selected trusted examples ACC: {} (Num: {})".format(Train_data_ACC_1, Train_data_num_1))
    alg_options['train_dataset'].update_trusted_label(-1 * np.ones(N), range(N), model_idx="1")
    alg_options['train_dataset'].update_trusted_label(trusted_labels_1, trusted_idx_1, model_idx="1")


    # model 2
    trusted_labels_true_2 = alg_options['train_dataset'].train_labels[trusted_idx_2]
    Train_data_num_2 = len(trusted_idx_2)
    Train_data_ACC_2 = (np.array(trusted_labels_2) == np.array(trusted_labels_true_2)).sum() / Train_data_num_2

    logger.info("Model 2 (train) -- Selected trusted examples ACC: {} (Num: {})".format(Train_data_ACC_2, Train_data_num_2))
    alg_options['train_dataset'].update_trusted_label(-1 * np.ones(N), range(N), model_idx="2")
    alg_options['train_dataset'].update_trusted_label(trusted_labels_2, trusted_idx_2, model_idx="2")

    return 