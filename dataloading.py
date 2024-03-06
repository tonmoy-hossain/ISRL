import SimpleITK as sitk
import os, glob
import sys
from tqdm.notebook import trange, tqdm

import numpy as np
from numpy import zeros, newaxis
from numpy.random import seed
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


from glob import glob
import json

import pickle
import pandas as pd
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader
import torch

# datapath = ''


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

def normalize_img(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = img.astype('float32')
    return img


ENV_C_RANGES = [(50, 69), (70, 79), (80, 100)]
def get_age_env(age):
    for idx, (low, high) in enumerate(ENV_C_RANGES):
        if age >= low and age <= high:
            return idx
        
        
class ADNI_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels_names_ages):
        'Initialization'
        self.labels_names_ages = labels_names_ages
        self.image_paths = image_paths

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        paths = self.image_paths[index]
        y = self.labels_names_ages[index]
        
        X_channels = []
        
        for p in paths: # filename_src, filename_tar OR filename_image, filename_vel...
            itkimage = sitk.ReadImage(p)
            img = sitk.GetArrayFromImage(itkimage)
            img = normalize_img(img)
            if len(img.shape) == 3: # 3D no channel 
                img = img[newaxis,:,:]
            X_channels.append(img)
            
        X = np.concatenate(X_channels)

        label = y[0]
        label = 1 if y[0] == 'CN' else 0
        name = y[1]
        age = y[2]
        env = get_age_env(age) # get env index instead of age e.g. 60 = 0, 73 = 1, 85 = 2
        
        return X, label, name, env
    

    
class MNIST_Dataset_Direct(torch.utils.data.Dataset):
    def __init__(self, images, labels_names_colors):
        'Initialization'
        self.labels_names_colors = labels_names_colors
        self.images = images

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.images[index]
        y = self.labels_names_colors[index]

        label = y[0]
        name = y[1]
        env = y[2] # env = color

        return X, label, name, env
    

class MNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels_names_colors, channels, dims):
        'Initialization'
        self.labels_names_colors = labels_names_colors
        self.image_paths = image_paths
        self.channels = channels
        self.dims = dims

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        paths = self.image_paths[index]
        y = self.labels_names_colors[index]
        label = y[0]
        name = y[1]
        env = y[2] # env = color
        
        X_channels = []

        for p in paths: # filename_src, filename_tar OR filename_image, filename_vel...
            itkimage = sitk.ReadImage(p)
            img = sitk.GetArrayFromImage(itkimage)
            img = normalize_img(img)
    #             img = resize(img, (dim, dim))
            X_channels.append(np.expand_dims(img, 0)) # bring channel forward, add 1 for concatenate so = (1,3,128,128)

        X = torch.zeros(2, self.channels, self.dims[0], self.dims[1]) # color source same as target
        X[:,env,:,:] = torch.tensor(np.concatenate(X_channels))
        
#         X_channels = []
#         for p in paths: # filename_src, filename_tar OR filename_image, filename_vel...
#             itkimage = sitk.ReadImage(p)
#             img = sitk.GetArrayFromImage(itkimage)
#             img = normalize_img(img)
#             X_channels.append(np.expand_dims(img.transpose(2,0,1),0))
#         X = np.concatenate(X_channels)

        return X, label, name, env
    

def visualize_dataloader(loader):
    ## visualize
    for j, (X, label, name, env) in enumerate(loader):
        inputs = X.to(dev)
        print(inputs.shape)
        label = label.to(dev)

        num_channels = inputs.shape[1]
        print('num_channels=',num_channels)

        for i in range(0, 4, 1):
            print('AD' if label[i].cpu().numpy() == 0 else 'CN', name[i].split('.')[0], env[i].cpu().numpy())

            plt.figure(figsize=(8,3))

            for c in range(num_channels):
                plt.subplot(1,num_channels,c+1)
                plt.imshow(inputs[i,c,:,:,:].squeeze().cpu().numpy()[:,64,:], cmap='gray')
                plt.axis('off')
            plt.show()

        break
        

def visualize_2d_dataloader(loader):
    ## visualize
    for j, (inputs, labels, name, env) in enumerate(loader):
        num_channels = inputs.shape[1]
        print('num_channels=',num_channels)

        for i in range(0, 5, 1):
            print(inputs[i].shape)
            print(labels[i].detach().cpu().numpy(), name[i], env[i].detach().cpu().numpy())
            plt.figure(figsize=(3,3))
            
            for c in range(num_channels):
                plt.subplot(1,num_channels,c+1)
                plt.imshow(inputs[i,c,].squeeze().cpu().numpy().transpose(1,2,0))
                plt.axis('off')
            plt.show()
        break

##################Data Loading##########################
def get_adni_traintest_reg_data(only_env=None, limit=None):
    data = {
        'train': [],
        'train_labels_names_ages': [],
        'test': [],
        'test_labels_names_ages': [],
        'val': [],
        'val_labels_names_ages': [],
    }

    datapath = ''
    
    for keyword in ['train', 'test', 'val']:
        # readfilename = 'ADNI_TR_TE/ADNI_mixed_train.json'
        readfilename = os.path.join(datapath, f'ADNI_{keyword}_450.json')
        json_data = json.load(open(readfilename, 'r'))
        outputs = []
        labels_names_ages = []

        data_csv_path = os.path.join(datapath, f'ADNI_{keyword[:2].upper()}.csv') # ADNI_TR.csv / ADNI_TE.csv
        df = pd.read_csv(data_csv_path)
        df = df.set_index('Image Data ID') # dataframe of image data to read ages
        
        for i in range(len(json_data[keyword])):
            fpath_src = os.path.join(datapath, json_data[keyword][i]['source_img'])
            fpath_tar = os.path.join(datapath, json_data[keyword][i]['target_img'])
        
            label = fpath_tar.split('/')[-2]
            name = fpath_tar.split('/')[-1].split('.')[0]
            age = df.loc[name]['Age'] # read age
            env = get_age_env(age)
            
            if only_env is not None and env != only_env:
                continue

            paths = np.concatenate(([fpath_src], [fpath_tar]))
            outputs.append(paths)
            labels_names_ages.append((label, name, age))
            
            if limit is not None and len(outputs) == limit:
                break

        data[keyword] = outputs
        data[keyword+'_labels_names_ages'] = labels_names_ages

        print(len(data[keyword]), len(data[keyword+'_labels_names_ages']))
    
    return data



def get_adni_traintest_image_data_direct(only_env=None):
    data = {
        'train': [],
        'train_labels_names_ages': [],
        'test': [],
        'test_labels_names_ages': [],
        'val': [],
        'val_labels_names_ages': [],
    }
    
    datapath = 'ADNI_TR_TE'
    
    for keyword in ['train', 'test', 'val']:
        outputs = []
        labels_names_ages = []
        
        datapath = 'ADNI_TR_TE'
        data_csv_path = os.path.join(datapath, f'ADNI_{keyword[:2].upper()}.csv') # ADNI_TR.csv / ADNI_TE.csv
        df = pd.read_csv(data_csv_path)
        df = df.set_index('Image Data ID') # dataframe of image data to read ages

        for class_name in ['AD', 'CN']:
            for filename in (os.listdir(os.path.join(datapath, keyword, class_name))):    
                fpath = os.path.join(datapath, keyword, class_name, filename)
                label = fpath.split('/')[-2]
                name = fpath.split('/')[-1].split('.')[0]
                age = df.loc[name]['Age'] # read age
                
                env = get_age_env(age)
                if only_env is not None and env != only_env:
                        continue
                
                paths = [fpath]
                outputs.append(paths)

                labels_names_ages.append((label, name, age))

        data[keyword] = outputs
        data[keyword+'_labels_names_ages'] = labels_names_ages

        print(len(data[keyword]), len(data[keyword+'_labels_names_ages']))    

    return data
    
    
    
def get_adni_traintest_imagevel_data(modalities, only_env=None, limit=None): # modalities = image, vel, imagevel
    data = {
        'train': [],
        'train_labels_names_ages': [],
        'test': [],
        'test_labels_names_ages': [],
    }
    
    datapath = 'ADNI_TR_TE'
    
    for keyword in ['train', 'test', 'val']:
        outputs = []
        labels_names_ages = []
        
        main_datapath = 'ADNI_TR_TE'
        data_csv_path = os.path.join(main_datapath, f'ADNI_{keyword[:2].upper()}.csv') # ADNI_TR.csv / ADNI_TE.csv
        df = pd.read_csv(data_csv_path)
        df = df.set_index('Image Data ID') # dataframe of image data to read ages
        
        datapath = f'{keyword}_result'
        datapath_ti = os.path.join(datapath, 'TI')
        datapath_vel = os.path.join(datapath, 'vel_0')
        outputs = []
        labels_names_ages = []
        
        for fname_tar in os.listdir(datapath_ti):
            if fname_tar.endswith('.mhd'):
                fname_vel = fname_tar.replace('tar_img', 'src_lbl')
                
                fpath_tar = os.path.join(datapath_ti, fname_tar)
                fpath_vel = os.path.join(datapath_vel, fname_vel)
                
                _, _, label, name = fname_tar.split('_')
                name = name.split('.')[0] # remove .mhd
                label = 'AD' if label == '0' else 'CN'
                
                age = df.loc[name]['Age'] # read age
                
                env = get_age_env(age)
                if only_env is not None and env != only_env:
                        continue
                                
                paths = []
                if modalities == 'image':
                    paths = [fpath_tar]
                elif modalities == 'vel':
                    paths = [fpath_vel]
                elif modalities == 'imagevel':
                    paths = [fpath_tar, fpath_vel]
                
                outputs.append(paths)
                labels_names_ages.append((label, name, age))
                
                if limit is not None and len(outputs) == limit:
                    break
                    
        data[keyword] = outputs
        data[keyword+'_labels_names_ages'] = labels_names_ages

        print(len(data[keyword]), len(data[keyword+'_labels_names_ages']))
    
    return data



def get_mnist_traintest_reg_data(only_env=None, limit=None):
    data = {
        'train': [],
        'train_labels_names_ages': [],
        'test': [],
        'test_labels_names_ages': [],
        'val': [],
        'val_labels_names_ages': [],
    }

    datapath = 'datasets/mnist'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    for keyword in ['train', 'test', 'val']:
        # readfilename = 'ADNI_TR_TE/ADNI_mixed_train.json'
        readfilename = os.path.join(datapath, f'mnist_{keyword}.json')
        json_data = json.load(open(readfilename, 'r'))
        outputs = []
        labels_names_colors = []

        for i in range(len(json_data[keyword])):
            fpath_src = json_data[keyword][i]['source_img']
            fpath_tar = json_data[keyword][i]['target_img']

            label = fpath_tar.split('/')[-2]
                
            name = fpath_tar.split('/')[-1].split('.')[0]
            color = int(name.split('_')[1])
            env = color
            
            if only_env is not None and env != only_env:
                continue

            paths = np.concatenate(([fpath_src], [fpath_tar]))
            outputs.append(paths)
            labels_names_colors.append((int(label), name, color))
            
            if limit is not None and len(outputs) == limit:
                break

        data[keyword] = outputs
        data[keyword+'_labels_names_colors'] = labels_names_colors

        print(len(data[keyword]), len(data[keyword+'_labels_names_colors']))
    
    return data



def get_mnist_traintest_reg_data(only_env=None, limit=None):
    data = {}

    datapath = 'datasets/quickdraw'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
#     for keyword in ['train', 'test', 'val']:
    for keyword in ['train1', 'train2', 'test', 'val']:
        print('reading from', keyword)
        data[keyword] = []
        data[f'{keyword}_labels_names_colors'] = []
        
        # readfilename = 'ADNI_TR_TE/ADNI_mixed_train.json'
        readfilename = os.path.join(datapath, f'quickdraw_{keyword}.json')
        json_data = json.load(open(readfilename, 'r'))
        outputs = []
        labels_names_colors = []

        for i in range(len(json_data[keyword])):
            fpath_src = json_data[keyword][i]['source_img']
            fpath_tar = json_data[keyword][i]['target_img']
            label = fpath_tar.split('/')[-2]
            name = fpath_tar.split('/')[-1].split('.')[0]
            color = int(name.split('_')[1])
            env = color
            
            if only_env is not None and env != only_env:
                continue

            paths = np.concatenate(([fpath_src], [fpath_tar]))
            outputs.append(paths)
            labels_names_colors.append((int(label), name, color))
            
            if limit is not None and len(outputs) == limit:
                break

        data[keyword] = outputs
        data[keyword+'_labels_names_colors'] = labels_names_colors

        print(len(data[keyword]), len(data[keyword+'_labels_names_colors']))
    
    return data


def get_mnist_traintest_image_data_direct(only_env=None):
    data = {}
    
    datapath = 'datasets/mnist'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    for keyword in ['train', 'test', 'val']:
        data[keyword] = []
        data[f'{keyword}_labels_names_colors'] = []
        
        outputs = []
        labels_names_colors = []
        
        for class_name in classes:
            for filename in (os.listdir(os.path.join(datapath, keyword, class_name))):    
                if not filename.endswith('.png'):
                    continue
                fpath = os.path.join(datapath, keyword, class_name, filename)
                label = fpath.split('/')[-2]
                name = fpath.split('/')[-1].split('.')[0]
                color = int(name.split('_')[1])
                env = color

                if only_env is not None and env != only_env:
                        continue
                
                paths = [fpath]
                outputs.append(paths)

                labels_names_colors.append((int(label), name, color))

        data[keyword] = outputs
        data[keyword+'_labels_names_colors'] = labels_names_colors

        print(len(data[keyword]), len(data[keyword+'_labels_names_colors']))    

    return data