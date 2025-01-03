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

import numpy as np
from numpy.random import default_rng
import random
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

from itertools import chain

from utils.synthetic import *
from utils.tools import *
from Data_load.cifar10 import *
from Data_load.cifar100 import *
from Data_load.labelme import *
from Data_load.transformer import *
from models import LeNet, ResNet, VGG, Transition, FCNN
from torchvision.models import vgg19_bn

from train.train_tools import get_transition_a, get_transition_b, update_trusted_loss, update_trusted_acc

def CDRP(args,alg_options,logger, t): # (AdaptCDRP)
    
    logger = logging.getLogger()

    train_loader = alg_options['train_loader']
    val_loader = alg_options['val_loader']
    test_loader = alg_options["test_loader"]

    Num_train = alg_options['train_dataset'].__len__()
    Num_val = alg_options['val_dataset'].__len__()
    Num_classes = args.num_classes
    Num_annotator = args.R

    device = alg_options['device']

    # models

    if args.dataset == "cifar10":
        model_1 = ResNet.ResNet18(Num_classes)
        model_2 = ResNet.ResNet18(Num_classes)

    elif args.dataset == "cifar100":
        model_1 = ResNet.ResNet34(Num_classes)
        model_2 = ResNet.ResNet34(Num_classes)

    elif args.dataset == "labelme":
        model_1 = FCNN.FCNN()
        model_2 = FCNN.FCNN()
    
    elif args.dataset == "animal":
        model_1 = vgg19_bn(pretrained=False)
        model_1.classifier._modules['6'] = nn.Linear(4096, 10)
        model_2 = vgg19_bn(pretrained=False)
        model_2.classifier._modules['6'] = nn.Linear(4096, 10)

    else:
        logger.info('Incorrect choice for dataset')
    
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler_1 = optim.lr_scheduler.OneCycleLR(optimizer_1, args.learning_rate, epochs=args.n_epoch, steps_per_epoch=len(train_loader), verbose=True)

    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler_2 = optim.lr_scheduler.OneCycleLR(optimizer_2, args.learning_rate, epochs=args.n_epoch, steps_per_epoch=len(train_loader), verbose=True)


    Loss = Loss_CDRP(eps=args.eps)

    Returned_Train_ACC_1 = []
    Returned_Train_LOSS_1 = []
    Returned_Val_ACC_1 = []
    Returned_Test_ACC_1 = []
    Returned_Train_ACC_2 = []
    Returned_Train_LOSS_2 = []
    Returned_Val_ACC_2 = []
    Returned_Test_ACC_2 = []
    Returned_Train_ESTIMATION_ERROR = []

    best_val_acc_1 = 0
    best_val_acc_2 = 0
    test_acc_on_best_model_1 = 0
    test_acc_on_best_model_2 = 0
    
    error_rate = args.error_rate
    trusted_ratio = np.ones(args.n_epoch) * (1 - error_rate)
    trusted_ratio[ :int(args.n_epoch/2)] = np.linspace(0, 1 - error_rate, int(args.n_epoch/2))

    loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean')

    if os.path.exists(args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_model1_warmup.pth'):
        print("Exist warmed up models!")
        epoch_range = range(args.n_epoch_burn, args.n_epoch)
    else:
        epoch_range = range(args.n_epoch)


    for epoch in epoch_range:


        ############################### Train ###############################

        logger.info('')
        logger.info('##############################[Epoch {}]################################'.format(epoch+1))
        logger.info('')
        logger.info("[Epoch {}] Training model -- CDRP...".format(epoch+1))

        train_total_loss_1 = 0
        train_total_loss_2 = 0
        train_total_correct_1 = 0 
        train_total_correct_2 = 0 
        train_total_post_sel_correct_1 = 0 
        train_total_post_sel_correct_2 = 0 
        train_num = 0

        total_batch = len(train_loader)

        model_1.train()
        model_2.train()

        if epoch >= args.n_epoch_burn:

            if epoch == args.n_epoch_burn:

                model_1.load_state_dict(torch.load(args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_model1_warmup.pth'))
                model_2.load_state_dict(torch.load(args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_model2_warmup.pth'))
            
            if (epoch == args.n_epoch_burn) or ((args.update_transition == "true") and ((epoch - args.n_epoch_burn) % args.es_T_update_freq == 0) and (epoch < args.n_epoch_burn/2)):

                model_1.eval()
                model_2.eval()

                logger.info("*---*---*---*---* Update transition matrices *---*---*---*---*")

                threshold = min(args.thr + (epoch - args.n_epoch_burn) * 0.01, 0.7)
                update_trusted_acc(model_1, model_2, alg_options, logger, threshold, device)

                if args.transition_type == "independent":
                    est_T, estimation_error = get_transition_a(args, alg_options, logger) # est_T: matrix
                    Returned_Train_ESTIMATION_ERROR.append(estimation_error)
                    np.save(args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_TransitionMatrix_CDRP.npy', est_T)

                else:
                    est_T, estimation_error = get_transition_b(args, alg_options, logger, t, model_1) # est_T: trained model
                    Returned_Train_ESTIMATION_ERROR.append(estimation_error) 
            
            logger.info("*---*---*---*---* Update trusted labels (small loss) *---*---*---*---*")
            
            threshold = trusted_ratio[epoch]
            update_trusted_loss(model_1, model_2, alg_options, logger, threshold, device)

            post_est_1 = torch.zeros((Num_train, Num_classes)).to(device)
            post_est_2 = torch.zeros((Num_train, Num_classes)).to(device)


        
        for batch_idx, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

            batch_size = batch_x.shape[0]
            train_num += batch_x.shape[0]

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                noisy_label = noisy_label.to(device)
                annot = annot.to(device)
                trusted_label_1 = trusted_label_1.to(device)
                trusted_label_2 = trusted_label_2.to(device)
            
            # probs
            probs_1 = model_1(batch_x).softmax(-1)
            probs_2 = model_2(batch_x).softmax(-1)

            # ACC
            y_hat_1 = torch.max(probs_1, 1)[1]
            batch_correct_1 = (y_hat_1 == batch_y).sum()
            train_total_correct_1 += batch_correct_1

            y_hat_2 = torch.max(probs_2, 1)[1]
            batch_correct_2 = (y_hat_2 == batch_y).sum()
            train_total_correct_2 += batch_correct_2

            # -------- Train --------

            if epoch < args.n_epoch_burn:

                # warm up: 

                loss_1 = loss_fn(torch.log(probs_1 + 1e-10), noisy_label)
                batch_loss_1 = loss_1.item()
                train_total_loss_1 += batch_loss_1 * batch_size

                loss_2 = loss_fn(torch.log(probs_2 + 1e-10), noisy_label)
                batch_loss_2 = loss_2.item()
                train_total_loss_2 += batch_loss_2 * batch_size

                optimizer_1.zero_grad()
                loss_1.backward()
                optimizer_1.step()

                optimizer_2.zero_grad()
                loss_2.backward()
                optimizer_2.step()

            else:

                # -------- Train with AdaptCDRP --------

                # prior: model output P(y|x)

                model_1.train()
                model_2.train()

                prior_1 = model_1(batch_x).softmax(-1)
                prior_2 = model_2(batch_x).softmax(-1) 

                # p(y~|y,x) (y~ denotes noisy annotation)

                if args.transition_type == "independent":
                    batch_T = est_T.unsqueeze(0).to(device) 
                else:
                    batch_T = est_T(batch_x).to(device)
                
                if args.one_transition == "true":
                    batch_T = batch_T.expand(batch_size, 1, Num_classes, Num_classes) 
                    noisy_one_hot = F.one_hot(noisy_label, Num_classes).float().unsqueeze(1) 
                    cond = torch.pow(batch_T, noisy_one_hot.float().unsqueeze(2)).prod(3).prod(1)

                else:
                    batch_T = batch_T.expand(batch_size, Num_annotator, Num_classes, Num_classes) 
                    annot_one_hot = torch.zeros((batch_size, Num_annotator, Num_classes)).to(device) 
                    annot_one_hot = annot_one_hot.view(batch_size * Num_annotator, Num_classes) 
                    mask = (annot.view(-1) != -1).to(device) # (n * R, )
                    annot_one_hot[mask] = F.one_hot(annot.view(-1)[mask], Num_classes).float()
                    annot_one_hot = annot_one_hot.view(batch_size, Num_annotator, Num_classes) 
                    cond = torch.pow(batch_T, annot_one_hot.float().unsqueeze(2)).prod(3).prod(1)
                
                # estimated true label posterior: p(y|y~, x) proportional to P(y|x) * p(y~|y,x)
                
                post_1_temp = prior_1 * cond
                post_1_temp /= post_1_temp.sum(1).unsqueeze(1)

                post_2_temp = prior_2 * cond
                post_2_temp /= post_2_temp.sum(1).unsqueeze(1)

                post_est_1[indexes] = post_1_temp.detach()
                post_est_2[indexes] = post_2_temp.detach()

                ## Theorem 3.1: robust pseudo-labels => pseudo-empirical distribution

                post_values_1, post_idx_1 = torch.topk(post_1_temp.detach(), k=2, dim=1)
                post_values_2, post_idx_2 = torch.topk(post_2_temp.detach(), k=2, dim=1)

                y_post_1 = post_idx_1[:, 0].detach() # label for max post_1_temp; use it to update model 2
                y_post_2 = post_idx_2[:, 0].detach() # label for max post_2_temp; use it to update model 1

                mask_thr = args.LRT_init + (epoch - args.n_epoch_burn) * args.LRT_incre
                mask_1 = (post_values_1[:, 0] / post_values_1[:, 1] > mask_thr) # use it to update model 2
                mask_2 = (post_values_2[:, 0] / post_values_2[:, 1] > mask_thr) # use it to update model 1

                batch_post_sel_correct_1 = (y_post_1[mask_1] == batch_y[mask_1]).sum() 
                train_total_post_sel_correct_1 += batch_post_sel_correct_1

                batch_post_sel_correct_2 = (y_post_2[mask_2] == batch_y[mask_2]).sum() # selected post correct
                train_total_post_sel_correct_2 += batch_post_sel_correct_2

                ## Theorem 3.2: update the Lagrange multiplier & calculate the worst-case risk

                post_1 = F.one_hot(y_post_1, Num_classes).float()
                post_2 = F.one_hot(y_post_2, Num_classes).float()

                loss_1_cdrp, loss_2_cdrp = 0, 0
                if mask_1.sum().float()>0 and mask_2.sum().float()>0: 
                    loss_1_cdrp, loss_2_cdrp = Loss(prior_1[mask_2], prior_2[mask_1], post_1[mask_1], post_2[mask_2])

                loss_1 = loss_1_cdrp
                loss_2 = loss_2_cdrp

                batch_loss_1 = loss_1.item()
                train_total_loss_1 += batch_loss_1 *  batch_size 

                optimizer_1.zero_grad()
                loss_1.backward()
                optimizer_1.step()
                optimizer_1.zero_grad()

                batch_loss_2 = loss_2.item()
                train_total_loss_2 += batch_loss_2 *  batch_size 

                optimizer_2.zero_grad()
                loss_2.backward()
                optimizer_2.step()
                optimizer_2.zero_grad()

            # Print
            if (batch_idx + 1) % args.print_freq == 0:
                
                if epoch >= args.n_epoch_burn:
                    print('Epoch [%d], Iter [%d/%d], Training ACC_1: %.2F, Training ACC_2: %.2F, Loss_1: %.4f, Loss_2: %.4f'
                        % (epoch + 1, batch_idx + 1, total_batch, 100*batch_correct_1/batch_size, 100*batch_correct_2/batch_size, batch_loss_1, batch_loss_2))
                    print('======== Selected post ACC_1: {}, Selected post ACC_2: {}'.format(
                        100*batch_post_sel_correct_1/mask_1.sum().float(), 100*batch_post_sel_correct_2/mask_2.sum().float()))
                else:
                    print('Epoch [%d], Iter [%d/%d], Training ACC_1: %.2F, Training ACC_2: %.2F, Loss_1: %.4f, Loss_2: %.4f'
                    % (epoch + 1, batch_idx + 1, total_batch, 100*batch_correct_1/batch_size, 100*batch_correct_2/batch_size, batch_loss_1, batch_loss_2))
        
        scheduler_1.step()
        scheduler_2.step()
        
        train_acc_1 = 100 * float(train_total_correct_1) / float(train_num)
        train_acc_2 = 100 * float(train_total_correct_2) / float(train_num)
        train_loss_1 = float(train_total_loss_1) / float(train_num)
        train_loss_2 = float(train_total_loss_2) / float(train_num)
    
        logger.info("Train ACC (model 1): {:.2f}; Train ACC (model 2): {:.2f}".format(train_acc_1, train_acc_2))
        logger.info("Train loss (model 1): {:.4f}; Train loss (model 2): {:.4f}".format(train_loss_1, train_loss_2))

        Returned_Train_ACC_1.append(train_acc_1)
        Returned_Train_ACC_2.append(train_acc_2)
        Returned_Train_LOSS_1.append(train_loss_1)
        Returned_Train_LOSS_2.append(train_loss_2)


        ############################### Validation ###############################
                
        logger.info('')
        logger.info("[Epoch {}] Validating model -- CDRP...".format(epoch+1))

        val_total_correct_1 = 0 # for true label
        val_total_correct_2 = 0 # for noisy label
        val_total_correct_1_ = 0 # for true label
        val_total_correct_2_ = 0 # for noisy label
        val_num = 0

        with torch.no_grad():
            model_1.eval() 
            model_2.eval() 

            for _, batch_x, batch_y, noisy_label in val_loader:

                val_num += batch_y.size(0)

                if torch.cuda.is_available:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    noisy_label = noisy_label.to(device)
                
                probs1 = model_1(batch_x).softmax(-1)
                y_hat_1 = torch.max(probs1, 1)[1]
                val_total_correct_1 += (y_hat_1 == batch_y).sum()
                val_total_correct_1_ += (y_hat_1 == noisy_label).sum()

                probs2 = model_2(batch_x).softmax(-1)
                y_hat_2 = torch.max(probs2, 1)[1]
                val_total_correct_2 += (y_hat_2 == batch_y).sum()
                val_total_correct_2_ += (y_hat_2 == noisy_label).sum()
            
        val_acc_1 = 100 * float(val_total_correct_1) / float(val_num) # true
        val_acc_2 = 100 * float(val_total_correct_2) / float(val_num)

        val_acc_1_ = 100 * float(val_total_correct_1_) / float(val_num) # noisy
        val_acc_2_ = 100 * float(val_total_correct_2_) / float(val_num)

        logger.info("Val ACC (model 1 -- true label): {:.4f}".format(val_acc_1))
        logger.info("Val ACC (model 2 -- true label): {:.4f}".format(val_acc_2))
        logger.info("Val ACC (model 1 -- noisy label): {:.4f}".format(val_acc_1_))
        logger.info("Val ACC (model 2 -- noisy label): {:.4f}".format(val_acc_2_))
        
        Returned_Val_ACC_1.append(val_acc_1)
        Returned_Val_ACC_2.append(val_acc_2)
                


        ################################## Test ##################################

        logger.info('')
        logger.info("[Epoch {}] Testing model -- CDRP...".format(epoch+1))

        test_total_correct_1 = 0 # for true label
        test_total_correct_2 = 0
        test_num = 0

        with torch.no_grad():
            model_1.eval()  
            model_2.eval() 

            for batch_x, batch_y in test_loader:

                test_num += batch_y.size(0)

                if torch.cuda.is_available:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                
                probs1 = model_1(batch_x).softmax(-1)
                y_hat_1 = torch.max(probs1, 1)[1]
                test_total_correct_1 += (y_hat_1 == batch_y).sum()

                probs2 = model_2(batch_x).softmax(-1)
                y_hat_2 = torch.max(probs2, 1)[1]
                test_total_correct_2 += (y_hat_2 == batch_y).sum()


        test_acc_1 = 100 * float(test_total_correct_1) / float(test_num)
        test_acc_2 = 100 * float(test_total_correct_2) / float(test_num)

        logger.info("Test ACC (model 1): {:.2f}".format(test_acc_1))
        logger.info("Test ACC (model 2): {:.2f}".format(test_acc_2))

        Returned_Test_ACC_1.append(test_acc_1)
        Returned_Test_ACC_2.append(test_acc_2)

        ############################### Model Selection: Validation ACC ###############################

        if val_acc_1_ > best_val_acc_1:
            best_val_acc_1 = val_acc_1_
            test_acc_on_best_model_1 = test_acc_1
            if epoch < args.n_epoch_burn:
                torch.save(model_1.state_dict(), args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_model1_warmup.pth')
            else:
                torch.save(model_1.state_dict(), args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_model1_CDRP.pth')
            print('Saved: model 1 of best val acc')
        
        if val_acc_2_ > best_val_acc_2:
            best_val_acc_2 = val_acc_2_
            test_acc_on_best_model_2 = test_acc_2
            if epoch < args.n_epoch_burn:
                torch.save(model_2.state_dict(), args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_model2_warmup.pth')
            else:
                torch.save(model_2.state_dict(), args.save_dir + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_model2_CDRP.pth')
            print('Saved: model 2 of best val acc')
        
        print("Best model ACC (test) -- model1:", test_acc_on_best_model_1)
        print("Best model ACC (test) -- model2:", test_acc_on_best_model_2)
    
    ############################################ Save and return result ############################################

    out = {}

    out["Train_ACC_1"] = Returned_Train_ACC_1
    out["Train_loss_1"] = Returned_Train_LOSS_1
    out["Val_ACC_1"] = Returned_Val_ACC_1
    out["Test_ACC_1"] = Returned_Test_ACC_1

    out["Train_ACC_2"] = Returned_Train_ACC_2
    out["Train_loss_2"] = Returned_Train_LOSS_2
    out["Val_ACC_2"] = Returned_Val_ACC_2
    out["Test_ACC_2"] = Returned_Test_ACC_2

    out['Train_ESTIMATION_ERROR'] = Returned_Train_ESTIMATION_ERROR

    np.save(args.save_dir  + 'eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_trial_' + str(t) + '_result_CDRP.npy', out)

    return test_acc_on_best_model_1, test_acc_on_best_model_2
        


            



######################################################################################################################################

class Loss_CDRP(nn.Module):
    def __init__(self, gamma=5.0, eta=0.5, eps=0.01): 
        super(Loss_CDRP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eps = eps
        self.gamma_1 = torch.tensor(gamma).to(self.device)
        self.gamma_2 = torch.tensor(gamma).to(self.device)
        self.eta_1 = torch.tensor(eta).to(self.device)
        self.eta_2 = torch.tensor(eta).to(self.device)
    
    def forward(self,prior_1, prior_2, post_1, post_2):

        loss_fn = nn.NLLLoss(ignore_index=-1, reduction='none') 

        # loss_1 
        N = prior_1.shape[0]
        K = prior_1.shape[1]

        probs_1_expand = torch.clamp(torch.cat([prior_1 for i in range(K)], dim=0), min=self.eps, max=1-self.eps).to(self.device) # (N * K, K)
        y_expand = torch.cat([torch.ones(N) * i for i in range(K)], 0).to(self.device) # (N * K, )

        loss_temp_0 = self.eps * torch.clamp(self.gamma_1, min=0)
        loss_temp_1 = loss_fn(torch.log(probs_1_expand + 1e-10), y_expand.long()).reshape(K, N).transpose(0, 1) # l(psi(x),y'): (N * K, ) -> (N, K)
        loss_temp_2 = torch.clamp(self.gamma_1, min=0) * (torch.ones((K, K)) - torch.eye(K)).to(self.device) # c(y', Y): (K, K)
        loss_temp_3 = loss_temp_1.unsqueeze(1) - loss_temp_2.unsqueeze(0) # (N, 1, K) - (1, K, K) -> (N, K, K)
        loss_temp_4, idx = torch.max(loss_temp_3, 2)

        loss_1 = loss_temp_0 + torch.dot(post_2.reshape(-1).detach().to(self.device), loss_temp_4.reshape(-1)) / N

        # update gamma
        loss_max = torch.max(loss_temp_1.detach(), dim=1).values # (N, )
        alphas = (loss_max.unsqueeze(1) - loss_temp_1.detach()).reshape(-1) # (N, K) -> (N*K, )
        betas = post_2.detach().reshape(-1) # (N, K) -> (N*K, )
        alphas_sorted, indices = torch.sort(alphas, descending=True)
        betas_temp = torch.cumsum(betas.detach()[indices], dim=0) / (N*K) - self.eps
        betas_temp_sgn = torch.sign(betas_temp)
        if torch.all(betas_temp_sgn>=0): # args.eps <= betas[indices[0]] / (NK)
            s_star = 1
            gamma_10 = alphas_sorted[0]
            gamma_10 = gamma_10.detach()
        elif torch.all(betas_temp_sgn<=0): # args.eps >= betas.sum() / (NK)
            s_star = N + 1
            gamma_10 = 0
        else:
            s_star = ((betas_temp <= 0).nonzero())[-1]
            gamma_10 = alphas_sorted[s_star]
            gamma_10 = gamma_10.detach()
        
        self.gamma_1 = gamma_10 - self.eta_1 * (self.eps - ((idx == torch.cat([torch.ones((N, 1)) * i for i in range(K)], dim=1).to(self.device)) * post_2).sum() / N)
        
      
        # loss_2
        N = prior_2.shape[0]
        K = prior_2.shape[1]

        probs_2_expand = torch.clamp(torch.cat([prior_2 for i in range(K)], dim=0), min=self.eps, max=1-self.eps).to(self.device) # (N * K, K)
        y_expand = torch.cat([torch.ones(N) * i for i in range(K)], 0).to(self.device) # (N * K, )

        loss_temp_0 = self.eps * torch.clamp(self.gamma_2, min=0)
        loss_temp_1 = loss_fn(torch.log(probs_2_expand + 1e-10), y_expand.long()).reshape(K, N).transpose(0, 1) # l(psi(x),y'): (N * K, ) -> (N, K)
        loss_temp_2 = torch.clamp(self.gamma_2, min=0) * (torch.ones((K, K)) - torch.eye(K)).to(self.device) # c(y', Y): (K, K)
        loss_temp_3 = loss_temp_1.unsqueeze(1) - loss_temp_2.unsqueeze(0) # (N, 1, K) - (1, K, K) -> (N, K, K)
        loss_temp_4, idx = torch.max(loss_temp_3, 2)

        loss_2 = loss_temp_0 + torch.dot(post_1.reshape(-1).detach().to(self.device), loss_temp_4.reshape(-1)) / N

        # update gamma
        loss_max = torch.max(loss_temp_1.detach(), dim=1).values # (N, )
        alphas = (loss_max.unsqueeze(1) - loss_temp_1.detach()).reshape(-1) # (N, K) -> (N*K, )
        betas = post_1.detach().reshape(-1) # (N, K) -> (N*K, )
        alphas_sorted, indices = torch.sort(alphas, descending=True)
        betas_temp = torch.cumsum(betas.detach()[indices], dim=0) / (N*K) - self.eps
        betas_temp_sgn = torch.sign(betas_temp)
        if torch.all(betas_temp_sgn>=0): # args.eps <= betas[indices[0]] / (NK)
            s_star = 1
            gamma_20 = alphas_sorted[0]
            gamma_20 = gamma_20.detach()
        elif torch.all(betas_temp_sgn<=0): # args.eps >= betas.sum() / (NK)
            s_star = N + 1
            gamma_20 = 0
        else:
            s_star = ((betas_temp <= 0).nonzero())[-1]
            gamma_20 = alphas_sorted[s_star]
            gamma_20 = gamma_20.detach()

        self.gamma_2 = gamma_20 - self.eta_2 * (self.eps - ((idx == torch.cat([torch.ones((N, 1)) * i for i in range(K)], dim=1).to(self.device)) * post_1).sum() / N)


        return loss_1, loss_2
















    


