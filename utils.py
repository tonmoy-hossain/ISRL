import SimpleITK as sitk
import os, glob
import pickle
import torch,gc
from torch.autograd import grad
from torch.autograd import Variable
import torch.fft ############### Pytorch >= 1.8.0
import torch.nn.functional as F
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

def print_clear_gpu_ram():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                 print(obj.shape)
                del obj
                print(type(obj), obj.size())

        except:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model_save_path='.', patience=5, verbose=False, delta=0, paths=['checkpoint0.pt', 'checkpoint1.pt'], trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.paths = paths
        self.trace_func = trace_func
        self.model_save_path = model_save_path
        
        
    def __call__(self, val_loss, models):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models)
            self.counter = 0

    def save_checkpoint(self, val_loss, models):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        for i, model in enumerate(models):
            torch.save(model.state_dict(), os.path.join(self.model_save_path, self.paths[i]))
        
        self.val_loss_min = val_loss
        

        
def file_save(path, src_img, tar_img, def_img, vel, labels, names, b):  
    src_img = src_img.cpu().detach().numpy()
    tar_img = tar_img.cpu().detach().numpy()
    def_img = def_img.cpu().detach().numpy()
    vel = vel.cpu().detach().numpy()
    # def_seg_bin = def_seg_bin.round()

    for i in range(0, b, 1):
        img = sitk.GetImageFromArray(src_img[i].squeeze(), isVector=False)
        sitk.WriteImage(img, path + f"SI/src_img_{labels[i]}_{names[i]}.mhd", useCompression=False)

        img = sitk.GetImageFromArray(tar_img[i].squeeze(), isVector=False)
        sitk.WriteImage(img, path + f"TI/tar_img_{labels[i]}_{names[i]}.mhd", useCompression=False)
        
        img = sitk.GetImageFromArray(def_img[i].squeeze(), isVector=False)
        sitk.WriteImage(img, path + f"DI/def_img_{labels[i]}_{names[i]}.mhd", useCompression=False)

        img = sitk.GetImageFromArray(vel[i].squeeze(), isVector=False)
        sitk.WriteImage(img, path + f"vel_0/src_lbl_{labels[i]}_{names[i]}.mhd", useCompression=False)

        
def file_plot(src_img, tar_img, def_img, vel, labels, names, b):  
    src_img = src_img.cpu().detach().numpy()
    tar_img = tar_img.cpu().detach().numpy()
    def_img = def_img.cpu().detach().numpy()
    vel = vel.cpu().detach().numpy()
    # def_seg_bin = def_seg_bin.round()

    for i in range(0, b, 1):
        src = src_img[i].squeeze()
        tar = tar_img[i].squeeze()
        defo = def_img[i].squeeze()
        print(labels[i], names[i])
        
#         print(src.shape)
        plt.figure(figsize=(8,24))
        plt.subplot(1,3,1)
        plt.imshow(src[:,64,:])
        plt.subplot(1,3,2)
        plt.imshow(tar[:,64,:])
        plt.subplot(1,3,3)
        plt.imshow(defo[:,64,:])
        plt.show()