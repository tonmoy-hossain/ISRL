import torch
from torchvision import datasets
from torch import nn, optim, autograd
import torch.nn.functional as F
from matplotlib import pyplot as plt

import argparse
import numpy as np

from typing import Union, List, Tuple

import importlib
import pickle
import dataloading
importlib.reload(dataloading)

from dataloading import get_mnist_traintest_reg_data, normalize_img, MNIST_Dataset, MNIST_Dataset_Direct
from dataloading import ADNI_Dataset, get_adni_traintest_imagevel_data, get_adni_traintest_image_data_direct, get_adni_traintest_reg_data

import SimpleITK as sitk
from skimage.transform import resize
from utils import print_clear_gpu_ram

from sklearn.metrics import f1_score, auc, precision_score, recall_score, roc_auc_score, roc_curve, classification_report, accuracy_score, top_k_accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
import pandas as pd
from scipy.special import softmax

import models
importlib.reload(models)
from models import FCNModel, CNNModel, JointCNNModel
from network_epdiff_ISRL import loss_Reg, loss_Reg_2D, get_reg_net
import os


channels = 3
dim = 224
num_classes = 3
sigma = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


from sklearn.metrics import f1_score, auc, precision_score, recall_score, roc_auc_score, roc_curve, classification_report, accuracy_score, top_k_accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
import pandas as pd
from scipy.special import softmax


def evaluate_pred(y_true, y_pred, y_prob):
    print(len(y_true), len(y_pred), len(y_prob))
    print(y_true[:100])
    print(y_pred[:100])

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')
    acc_score = accuracy_score(y_true, y_pred)

    y_bin_prob = np.max(softmax(y_prob, axis=1), 1)
    
    mdf = pd.DataFrame([[acc_score, precision, recall, f1score]], columns=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
    print(mdf)
    
    return mdf, (y_true, y_pred, y_bin_prob)


def evaluate_reg_classifier(net, classifier, testloader, do_fuse=False, just_cnn=False, is_joint=False, grayscale_model=False):
    if net is not None:
        net.eval()
    classifier.eval()
    
    y_pred = []
    y_true = []
    y_prob = []

    with torch.no_grad():
        for j, (images, labels, name, env) in enumerate(testloader):
            images = images.to(dev)
            labels = labels.to(dev)
            b, m, c, w, h = images.shape # m for modalities - src/tar, c for rgb =3 
            src_img = images[:,0,...].reshape(b,c,w,h)
            tar_img = images[:,1,...].reshape(b,c,w,h)     
            
            if grayscale_model:
                tar_img = tar_img.sum(dim=1, keepdim=True)
                
            # =========== PRED =================
            if just_cnn: # only CNN
                pred_labels = classifier(tar_img)

            else: # joint / FCN with registration
                # convert_to_grayscale
                src_img_gray = src_img.sum(axis=1).reshape(b,1,w,h)
                tar_img_gray = tar_img.sum(axis=1).reshape(b,1,w,h)

                pred = net(src_img_gray, tar_img_gray, registration = True)
                reg_latent = pred[2]

                ################## REG ONLY (geo) vs JOINT (geo+intensity)
                if not is_joint:
                    pred_labels = classifier(reg_latent)
                else:
                    if do_fuse:
                        pred_labels = classifier((tar_img, reg_latent))
                    else:
                        pred_labels = classifier((tar_img, None))
                        
#             predicted = torch.max(pred_labels.data, 1)[1]
            predicted = torch.argmax(pred_labels.data, 1)
        
            y_prob += pred_labels.data.tolist()
            y_pred += predicted.data.tolist()
            y_true += labels.data.tolist()

    m, y = evaluate_pred(y_true, y_pred, y_prob)
    return m, y



##### Data Loading from Scratch for both 2D and 3D datasets ######
######## We already provided envs.pkl file under datasets.zip. Please use that to get a headstart ########

batch_size = 32
dev = 'cuda'

num_classes = 3
dim = 224
channel = 3

def read_images_path_list_coloring(path_list, labels_names_colors, add_grayscale_channel):
#     path_list, labels_names_colors = data[keyword], data[f'{keyword}_labels_names_colors']
    colors = np.array(labels_names_colors)[:,2].astype(int)
    
    channel = 3
    if add_grayscale_channel:
        channel = 4
    
    images = torch.zeros(len(path_list), 2, channel, dim, dim)
    for i in range(len(path_list)):
        paths = path_list[i]
        X_channels = []

        for p in paths: # filename_src, filename_tar OR filename_image, filename_vel...
            itkimage = sitk.ReadImage(p)
            img = sitk.GetArrayFromImage(itkimage)
            img = normalize_img(img)
    #             img = resize(img, (dim, dim))
            X_channels.append(np.expand_dims(img, 0)) # bring channel forward, add 1 for concatenate so = (1,3,128,128)

        X = torch.zeros(2, channel, dim, dim) # color source same as target
        X[:,colors[i],:,:] = torch.tensor(np.concatenate(X_channels))
        
        if add_grayscale_channel:
            grayscale = X.sum(axis=1)#.unsqueeze(dim=1)
#             print(X.shape, grayscale.shape)
            X[:,-1,:,:] = grayscale
            
        images[i] = X

    return images

def create_env_loaders(add_grayscale_channel=False):
    envs = []
    
    data = get_mnist_traintest_reg_data()
    
    for keyword in ['train1', 'train2', 'test']:
        print(keyword)
        images = read_images_path_list_coloring(data[keyword], data[f'{keyword}_labels_names_colors'], add_grayscale_channel)        
        loader = torch.utils.data.DataLoader(MNIST_Dataset_Direct(images=images, labels_names_colors=data[f'{keyword}_labels_names_colors']), batch_size=batch_size, shuffle=False, num_workers=0)
    
#         mnist_dataset = MNIST_Dataset(image_paths=data[keyword], labels_names_colors=data[f'{keyword}_labels_names_colors'], channels=num_classes, dims=(dim, dim))
#         loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
        envs.append({'loader':loader})
        
    return envs

def create_env_loaders_adni(modalities='image', is_joint=False):
    envs = []
    for e in [0, 1, 2]:    
        if is_joint:
            data = get_adni_traintest_reg_data(only_env=e)
        else:
            data = get_adni_traintest_image_data_direct(only_env=e)
        
        clf_trainloader_env = torch.utils.data.DataLoader(ADNI_Dataset(image_paths=data['train'], 
                                                                       labels_names_ages=data['train_labels_names_ages']), 
                                                          batch_size=batch_size, shuffle=False, num_workers=0)
        envs.append({'loader':clf_trainloader_env})
    
    
    if is_joint:
        data = get_adni_traintest_reg_data()
    else:
        data = get_adni_traintest_image_data_direct()
    
    clf_testloader_env = torch.utils.data.DataLoader(ADNI_Dataset(image_paths=data['test'], 
                                                                  labels_names_ages=data['test_labels_names_ages']), shuffle=False, 
                                                     batch_size=batch_size, num_workers=0)
    
    envs.append({'loader':clf_testloader_env})

    return envs

# envs = create_env_loaders_adni()    

# print('\nPrepared environments:')
# for env in envs:
#     print(len(env['loader'])*batch_size)

# envs = create_env_loaders()    

# print('\nPrepared environments:')
# for env in envs:
#     print(len(env['loader'])*batch_size)


# Save variable data to a file
# with open("/scratch/pwg7jb/envs_3k.pkl", "wb") as file:
#     pickle.dump(envs, file)

# Load variable data from the file
with open("/scratch/pwg7jb/envs_3k.pkl", "rb") as file:
    envs = pickle.load(file)
    

    
#### Pretty Print ####
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

            
def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))
    
    
   
##### ERM/IRM classes #####

class CNN_IRM: 
    def __init__(self, model, lr, wd, flags):
        self.flags = flags
        self.model = model
        self.optimizer_clf = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.grayscale_model = flags.grayscale_model
        
    def forward_pass(self, images, labels):
        images = images.to(dev)
        b, m, c, w, h = images.shape

        src_bch = images[:,0,...].reshape(b,c,w,h)
        tar_bch = images[:,1,...].reshape(b,c,w,h)        
        
        if self.grayscale_model:
            tar_bch = tar_bch.sum(dim=1, keepdim=True)
        
        logits = self.model(tar_bch)    
    
        loss = mean_nll(logits, labels) 
        
        return logits, loss
    
    def get_weight_norm(self):
        weight_norm = torch.tensor(0.).cuda()
        
        params = self.model.parameters()
        if flags.use_phi_params:
            params = self.model.classifier.parameters()
            
        for w in params: # change to clf params
            weight_norm += w.norm().pow(2)
        return weight_norm
    
    def get_models(self):
        return [self.model]
    
    def set_models_train(self):
        self.model.train()
    
    def set_models_eval(self):
        self.model.eval()
        
    def train(self, envs, epochs):
        pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

        num_batches = len(envs[0]['loader'])
        print('num_batches=', num_batches)
        print('penalty weight and penalty anneal iterations: ', self.flags.penalty_weight, self.flags.penalty_anneal_iters)

        
        for step in range(epochs): # epochs
            self.epoch = step

            for i in range(len(envs)):
                envs[i]['iter'] = iter(envs[i]['loader'])
            
            self.set_models_train()
            
            epoch_nll = 0
            epoch_pen = 0
            epoch_train_acc = 0
            epoch_test_acc = 0

            for j in range(num_batches): # load data from env's loader

                for i, env in enumerate(envs):
                    try:
                        images, orig_labels, names, ages = next(env['iter'])
                    except StopIteration:
                        continue
                        
                    labels = orig_labels.to(dev)
                    logits, total_loss = self.forward_pass(images, labels)

                    env['nll'] = total_loss
                    env['penalty'] = penalty(logits, labels)
        
                    env['acc'] = mean_accuracy(logits, labels) 
                
                train_nll = torch.stack([envs[i]['nll'] for i in range(flags.n_envs)]).mean()
                train_acc = torch.stack([envs[i]['acc'] for i in range(flags.n_envs)]).mean()
                train_penalty = torch.stack([envs[i]['penalty'] for i in range(flags.n_envs)]).mean()

            
                weight_norm = self.get_weight_norm()
                
                loss = train_nll.clone()
                loss += self.flags.l2_regularizer_weight * weight_norm # reg
                pw = (self.flags.penalty_weight if step >= self.flags.penalty_anneal_iters else 1.0)
                loss += pw * train_penalty # pen
                if pw > 1.0:
                  loss /= pw
                    

                self.optimizer_clf.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer_clf.step()
    
                test_acc = envs[-1]['acc']
        
                epoch_nll += train_nll.item()
                epoch_pen += train_penalty.item()
                epoch_train_acc += train_acc.item()
                epoch_test_acc += test_acc.item()
                

            if step % 1 == 0:
                print(np.int32(step),
                    epoch_nll / num_batches,
                    epoch_train_acc / num_batches,
                    epoch_pen / num_batches,
                    epoch_test_acc / num_batches)

        final_train_accs = (epoch_train_acc / num_batches)

        print('Final train acc:')
        print(np.mean(final_train_accs))
        
        final_test_accs = (epoch_test_acc / num_batches)

        print('Final test acc:')
        print(np.mean(final_test_accs))

        return final_train_accs



class Joint_IRM(CNN_IRM): 
    def __init__(self, net, classifier, lr, reg_error_func, alpha=0.1, beta=0.1, wd=0, flags=None): # reg net and classifier
        self.net = net
        self.classifier = classifier
        self.alpha = 0.1
        self.beta = 0.1
        self.flags = flags
        self.reg_error_func = reg_error_func
        self.optimizer_net, self.reg_criterion = optim.Adam(net.parameters(), lr=lr, weight_decay=wd), nn.MSELoss()
        params = list(net.parameters()) + list(classifier.parameters())
        self.optimizer_clf = optim.Adam(params, lr=lr, weight_decay=wd)
        
    def forward_pass(self, images, labels):
        images = images.to(dev)
        b, m, c, w, h = images.shape
        src_img = images[:,0,...].reshape(b,c,w,h)
        tar_img = images[:,1,...].reshape(b,c,w,h) 
        
        src_img_gray = src_img.sum(axis=1).reshape(b,1,w,h)
        tar_img_gray = tar_img.sum(axis=1).reshape(b,1,w,h)
        
        pred = self.net(src_img_gray, tar_img_gray, registration = True)

        reg_latent = pred[2]
        pred_label = self.classifier((tar_img, reg_latent))
        clf_loss = mean_nll(pred_label, labels)
        
        regularization = (pred[3]*pred[4]).sum() / (tar_img_gray.numel())
        reg_loss = self.reg_criterion(pred[0], tar_img_gray)/(sigma*sigma) + 0.01*regularization
        
        loss_total = clf_loss + reg_loss
#         if self.epoch >= self.flags.pre_train:
#             loss_total = reg_loss + clf_loss
#         else:
#             loss_total = reg_loss
            
        return pred_label, loss_total
    
    def get_weight_norm(self):
        weight_norm = torch.tensor(0.).cuda()
        
        for w in self.net.parameters():
            weight_norm += w.norm().pow(2)
            
        for w in self.classifier.parameters():
            weight_norm += w.norm().pow(2)
        
        return weight_norm
            
    def get_models(self):
        return [self.net, self.classifier]
    
    def set_models_train(self):
        self.net.train()
        self.classifier.train()
    
    def set_models_eval(self):
        self.net.eval()
        self.classifier.eval()
        
dims = (224, 224)
rnet = get_reg_net(dims=dims, nb_unet_features=[[16, 32, 32], [32, 32, 16, 16]])
reg_dim=25088

flags = {
        'dims': dims,
        'hidden_dim': 256,
        'n_restarts': 1,
        'l2_regularizer_weight': 0.00110794568,
        'penalty_anneal_iters': 5,
        'penalty_weight': 50000,
        'grayscale_model': False, ###############
        'steps': 501,   
        'num_classes': 3,
        'channels': 3,
        'n_envs': len(envs)-1,
        'use_phi_params': False,
        'is_2d': True,
    
        # for joint training
        'reg_dim': reg_dim,
        'pre_train': 5
}
flags = objectview(flags) 

def run_erm_irm(flags, erm, lr = 0.004898536566546834, epochs = 10, is_joint=False, encoder_type='custom'):
    
    print('grayscale_model: ', flags.grayscale_model)

    if erm:
        flags.penalty_anneal_iters = 0 
        flags.penalty_weight = 0
        print('Doing erm:', flags.penalty_anneal_iters, flags.penalty_weight)
    else:
        print('Doing IRM:', flags.penalty_anneal_iters, flags.penalty_weight, lr)
    
    print(epochs, lr)
    
    print('channels=', flags.channels)
    
    model_irm = None
    
    if is_joint:
        rnet = get_reg_net(flags.dims, [[16, 32, 32], [32, 32, 16, 16]])
        #     net = get_reg_net(dims, para.model.nb_unet_features)
        print(rnet.unet_model.encoder) # get shape here
        
        model = JointCNNModel(channels=flags.channels, dims=flags.dims, num_classes=flags.num_classes, op='cat', reg_dim=reg_dim, is_2d=flags.is_2d, encoder_type=encoder_type).to(dev) 
        
        model_irm = Joint_IRM(net=rnet, classifier=model, lr=lr, reg_error_func=loss_Reg_2D, wd=0, flags=flags)
    else:
    #     model = MLP(hidden_dim=flags.hidden_dim, n_classes=flags.num_classes, grayscale_model=flags.grayscale_model).to(dev)
        model = CNNModel(dims=flags.dims, channels=flags.channels, num_classes=flags.num_classes, is_2d=flags.is_2d, encoder_type=encoder_type).to(dev)  
        model_irm = CNN_IRM(model, lr=lr, wd=0, flags=flags)
    
    print(model)
    
    model_irm.train(envs, epochs)
    
    if is_joint:
        trained_rnet, trained_model = model_irm.get_models()
        return trained_rnet, trained_model
    else:
        trained_model = model_irm.get_models()[0]
        return trained_model


    
CUDA_LAUNCH_BLOCKING=1

metrics = []
lr = 0.00005
epochs = 15

model_name = 'resnet_irm'
encoder_type = 'resnet'

for i in range(100):
    torch.manual_seed(0)
    print('================================',i,'================================\n')
    model = run_erm_irm(flags, erm=False, lr=lr, epochs = epochs, encoder_type=encoder_type)
    m, p = evaluate_reg_classifier(None, model, envs[-1]['loader'], just_cnn=True, do_fuse=False, is_joint=False, grayscale_model=flags.grayscale_model)
    # rnet, model = run_erm_irm(flags, erm=True, lr=lr, epochs = epochs, is_joint=True, encoder_type=encoder_type)
    # m, p = evaluate_reg_classifier(rnet, model, envs[-1]['loader'], just_cnn=False, do_fuse=True, is_joint=True, grayscale_model=flags.grayscale_model)

    # torch.save(rnet, os.path.join('', str(i) + model_name+'_rnet.pth'))
    # torch.save(model, os.path.join('', str(i) + model_name+'_model.pth'))
    
    metrics.append(m)

print('lr: ', lr,  'epochs: ', epochs, 'penalties: ', flags.penalty_anneal_iters, flags.penalty_weight)
df_metrics = pd.concat(metrics, axis=1)
print('mean\n', df_metrics.mean(axis=1))
print('std\n', df_metrics.std(axis=1))
df_metrics
