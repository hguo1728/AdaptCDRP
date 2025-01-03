# import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import argparse

import numpy as np
from numpy.random import default_rng
import random
from sklearn.model_selection import train_test_split
from datetime import datetime
import time

from utils.synthetic import *
from Data_load.cifar10 import *
from Data_load.cifar100 import *
from Data_load.transformer import *
from Data_load.labelme import *
from Data_load.animal10n import animal10N_dataset, animal10N_test_dataset

from train import train_ours


parser = argparse.ArgumentParser()

########################################

parser.add_argument('--model_running', type = str, help='training methods', default='ours') 
parser.add_argument('--dataset',type=str,help='cifar10, cifar100, labelme, animal',default='cifar10')
parser.add_argument('--error_rate_type',type=str,help='error rates types: low, mid, high, real',default="low") 
parser.add_argument('--error_rate',type=float,help='error rate (or: estimate the error rate after warming up the models)',default=0.2)
parser.add_argument('--R',type=int,help='number of annotators: 5, 10, 30, 50, 100 (generate R annotators)',default=5) 
parser.add_argument('--l',type=int,help='number of annotations per instance: 1 (incomplete labeling: randomly select one annotation per instance)',default=1) 
parser.add_argument('--seed',type=int,help='Random seed: 1-5',default=1) 
parser.add_argument('--transition_type',type=str,help='transition matrix: instance independent/dependent',default="independent")  
parser.add_argument('--update_transition',type=str,help='update transition in the training process',default="false") 
parser.add_argument('--es_T_update_freq',type=int,help='frequency of updaing the transition matrices',default=10)
parser.add_argument('--one_transition',type=str,help='only get one transition matrix',default="false") 

# ----------------------------------------
parser.add_argument('--eps',type=float,help='epsilon \in (0, 1/K)',default=0.01) 
parser.add_argument('--LRT_init',type=float,help='initial threshold for post LRT: {10, 15, 20}',default=10.0) 
parser.add_argument('--LRT_incre',type=float,help='increment for post LRT threshold: {1.0, 1.5, 2.0}',default=1.0) 
parser.add_argument('--thr', type = float, help = 'threshold for collecting trusted examples (warm up): {0.4, 0.5}', default=0.5)



# ----------------------------------------

parser.add_argument('--save_dir',type=str,help='save directory',default=' ')
parser.add_argument('--device',type=int,help='GPU device number',default=0)
parser.add_argument('--n_trials',type=int,help='No of trials',default=5)
parser.add_argument('--print_freq', type=int, default=50)



######################################## 

# ----------------------------------------
parser.add_argument('--split_percentage', type = float, help = 'train and validation', default=0.9)
parser.add_argument('--norm_std', type = float, help = 'distribution ', default=0.1)
parser.add_argument('--num_classes', type = int, help = 'number of classes', default=10) 
parser.add_argument('--feature_size', type = int, help = 'input dimension', default=784) 

# ----------------------------------------
parser.add_argument('--num_workers', type = int, help='how many subprocesses to use for data loading', default=3)
parser.add_argument('--learning_rate',type=float,help='Learning rate',default=0.01)
parser.add_argument('--batch_size',type=int,help='Batch Size',default=128)
parser.add_argument('--n_epoch',type=int,help='Number of Epochs',default=100)
parser.add_argument('--n_epoch_burn',type=int,help='Number of Epochs (warm up)',default=20) 
parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4) 
parser.add_argument('--momentum', type=int, help='momentum', default=0.9) 

# ----------------------------------------
parser.add_argument('--learning_rate_T',type=float,help='transition matrix: Learning rate',default=0.001)
parser.add_argument('--n_epoch_T_init',type=int,help='transition matrix: number of epochs (train)',default=30)
parser.add_argument('--n_epoch_T_fine_tune',type=int,help='transition matrix: number of epochs (fine tune)',default=1)


args=parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Log file settings

if args.error_rate_type != "real":
    save_dir = 'result/' + args.model_running + '/' + args.dataset + '/' + args.error_rate_type + "_" + str(args.R) + "/" 
else:
    save_dir = 'result/' + args.model_running + '/' + args.dataset + '/' + args.error_rate_type + "/" 
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

args.save_dir = save_dir

time_now = datetime.now()
time_now.strftime("%b-%d-%Y")
log_file_name = save_dir + 'log_eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_' + str(args.seed) + '_' + str(time_now.strftime("%b-%d-%Y")) + '.txt'
result_file = save_dir + 'result_eps_' + str(args.eps) + '_LRT_' + str(args.LRT_init) + '_' + str(args.LRT_incre) + '_' + str(args.seed) + '_' + str(time_now.strftime("%b-%d-%Y")) + '.txt'

        
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')

fh = logging.FileHandler(log_file_name)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)




########################################################################################
#                                                                                      #
#                                  Load Dataset                                        #
#                                                                                      #
########################################################################################

def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    EM = False
    if args.model_running == 'EM':
        EM = True


    if args.dataset=='cifar10':
        train_data = cifar10_dataset(
                                train=True, 
                                transform=transform_train(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        val_data = cifar10_dataset(
                                train=False, 
                                transform=transform_test(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        test_data = cifar10_test_dataset(
                                transform=transform_test(args.dataset), 
                                target_transform=transform_target
                                     )
    
    if args.dataset=='cifar100':
        train_data = cifar100_dataset(
                                train=True, 
                                transform=transform_train(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        val_data = cifar100_dataset(
                                train=False, 
                                transform=transform_test(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger,num_class=args.num_classes, 
                                EM=EM
                                     )
        test_data = cifar100_test_dataset(
                                transform=transform_test(args.dataset), 
                                target_transform=transform_target
                                     )

    if args.dataset=='labelme':
        train_data = labelme_dataset(
                                train=True, 
                                transform=transform_train(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        val_data = labelme_dataset(
                                train=False, 
                                transform=transform_test(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        test_data = labelme_test_dataset(
                                transform=transform_test(args.dataset), 
                                target_transform=transform_target, test=True
                                     )
    
    if args.dataset=='animal':
        train_data = animal10N_dataset(
                                train=True, 
                                transform=None, target_transform=None, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        val_data = animal10N_dataset(
                                train=False, 
                                transform=None, target_transform=None, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        test_data = animal10N_test_dataset(
                                transform=None, 
                                target_transform=None
                                     )

    
    return train_data, val_data, test_data




########################################################################################
#                                                                                      #
#                                        Main                                          #
#                                                                                      #
########################################################################################


def main():

    ######################################## setup ########################################

    # Data logging variables
    
    fileid = open(result_file, "w")
    fileid.write('#########################################################\n')
    fileid.write(str(time_now))
    fileid.write('\n')
    fileid.write('Trial#\t')
    fileid.write(args.model_running)
    fileid.write('\n')

    if args.dataset == "cifar10":

        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.batch_size = 128
        args.learning_rate = 0.001
        args.n_epoch_burn = 30  # warm up
        args.n_epoch = 120 # train using AdaptCDRP: args.n_epoch - args.n_epoch_burn
        
        # ----- if using instance-dependent transition matrix: BayesianIDNT (Guo-Wang-Yi 2023.) -----

        args.n_epoch_T_init = 20 
        args.n_epoch_T_fine_tune = 10 
        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1        
        
    if args.dataset == "cifar100":

        args.num_classes = 100
        args.feature_size = 3 * 32 * 32
        args.batch_size = 128
        args.learning_rate = 0.001
        args.n_epoch_burn = 30  
        args.n_epoch = 150 
        
        # ----- if using instance-dependent transition matrix: BayesianIDNT (Guo-Wang-Yi 2023.) -----

        args.n_epoch_T_init = 20 
        args.n_epoch_T_fine_tune = 10 
        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1
    

    if args.dataset == "labelme":

        args.R = 59
        args.print_freq = 30
        args.num_classes = 8
        args.feature_size = 8192
        args.batch_size = 128
        args.learning_rate = 1e-2
        args.n_epoch_burn = 20 
        args.n_epoch = 100 
        

        # ----- if using instance-dependent transition matrix: BayesianIDNT (Guo-Wang-Yi 2023.) -----

        args.n_epoch_T_init = 20 
        args.n_epoch_T_fine_tune = 10 
        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1
    
    if args.dataset == "animal":

        args.R = 1
        args.num_classes = 10
        args.feature_size = 3 * 64 * 64
        args.batch_size = 128
        args.learning_rate = 0.1
        args.n_epoch_burn = 40
        args.n_epoch = 100 
        

        # ----- if using instance-dependent transition matrix: BayesianIDNT (Guo-Wang-Yi 2023.) -----

        args.n_epoch_T_init = 20 
        args.n_epoch_T_fine_tune = 10 
        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1

    ######################################## n_trials: repeated experiments ########################################
	
    t = args.seed

    logger.info(" ")
    logger.info('--*--*--*--*--*--*--*--*--*-- Starting trial ' + str(t) + '_' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + '--*--*--*--*--*--*--*--*--*--')
    logger.info(" ")

    # ------------------------------ Setup & Load Data ------------------------------

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Get the train, validation and test dataset 
    train_data, val_data, test_data = load_data(args)
    args.N = train_data.train_data.shape[0]
    
    # Prepare data for training/validation and testing
    train_loader = DataLoader(dataset=train_data,
                            batch_size=args.batch_size,
                            num_workers=3,
                            shuffle=True,
                            drop_last=False, 
                            pin_memory=False) 
    
    test_loader = DataLoader(dataset=test_data,
                            batch_size=args.batch_size,
                            num_workers=3,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False)
    
    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            num_workers=3,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False)
    
    alg_options = { 
        'device':device,
        'loss_function_type':'cross_entropy'}
    
    alg_options['train_dataset']=train_data
    alg_options['val_dataset']=val_data
    alg_options['annotators_sel']=range(args.R)
                            
    alg_options['train_loader'] = train_loader
    alg_options['val_loader'] = val_loader
    alg_options['test_loader']= test_loader
                        
    # ------------------------------ Run Algorithm ------------------------------

    print(args)

    fileid.write(str(args.seed)+'\t')

    logger.info('ours\n')
    test_acc_1, test_acc_2 = train_ours.CDRP(args,alg_options,logger, t)
    print("############### ACC_1 (ours): {} ###############".format(test_acc_1))
    fileid.write("%.4f\t" %(test_acc_1))
    fileid.write('\n')
    print("############### ACC_2 (ours): {} ###############".format(test_acc_2))
    fileid.write("%.4f\t" %(test_acc_2))
    fileid.write('\n')

    logger.removeHandler(fh)
    logger.removeHandler(ch)
			

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("Running time:", (t2-t1)/3600)
	

