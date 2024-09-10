import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from PIL import Image
from PIL import ImageFilter

import nibabel as nib
import SimpleITK as sitk
import pylab
from numpy import zeros, newaxis
from scipy import ndimage
import shutil


def color_grayscale_img(img, color):
    img = torch.stack([img,img,img],2)
        
    for c in range(3):
        if c != color:
            img[:,:,c] = 0
            
    return img

def save_torch_img(img, save_path):
    im = Image.fromarray((img.detach().cpu().numpy() * 255).astype(np.uint8))
#     im = im.filter(ImageFilter.SMOOTH)
    im = im.filter(ImageFilter.GaussianBlur(radius = 1))
    im = im.resize(newsize)
    im.save(save_path, dpi=(200, 200))
    

datapath = 'datasets/quickdraw'
newsize = (224, 224)

classes = [0, 1, 2]
lbl_map = {0:0, 1:1, 2:2}
class_count = np.zeros(3)

train = 700
val = 850
test = 1000

for keyword in ['train1', 'train2',  'val', 'test']:
    for c in classes:
        fpath = os.path.join(datapath, keyword, str(c))
        if not os.path.exists(fpath):
            os.mkdir(fpath)
            


rootdir = 'datasets/mhd'

data = []
targets = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        file_path = os.path.join(subdir, file)
        # print(file_path)
        if '.mhd' in file_path:
            img = sitk.ReadImage(file_path)
            img_array = sitk.GetArrayFromImage(img)
            data.append(img_array)
            if 'circle' in file_path:
                targets.append(0)
            elif 'square' in file_path:
                targets.append(1)
            else:
                targets.append(2) 

data = torch.Tensor(data)
targets = torch.Tensor(targets)

print(data.shape, targets.shape)


train_img = []
train_lbl = []

test_img = []
test_lbl = []

val_img = []
val_lbl = []

for i in range(targets.shape[0]):
    img = data[i].detach().cpu().numpy()
    img = (img - img.min())/(img.max() - img.min())
    lbl = int(targets[i].detach().cpu())

    if lbl in lbl_map:
        lbl = lbl_map[lbl]
        
        if class_count[lbl] < train:
            train_img.append(img)
            train_lbl.append(lbl)
            class_count[lbl] += 1
        elif class_count[lbl] < val:
            val_img.append(img)
            val_lbl.append(lbl)
            class_count[lbl] += 1
        elif class_count[lbl] < test:
            test_img.append(img)
            test_lbl.append(lbl)
            class_count[lbl] += 1

print(len(train_img), len(train_lbl), len(val_img), len(val_lbl), len(test_img), len(test_lbl))


train_img = torch.tensor(train_img)
test_img = torch.tensor(test_img)
val_img = torch.tensor(val_img)

train_lbl = torch.tensor(train_lbl)
test_lbl = torch.tensor(test_lbl)
val_lbl = torch.tensor(val_lbl)


def save_environment_multiclass(save_folder, images, labels, color_prob, label_prob, n_classes=2):
    print(save_folder)
    
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def collapse_labels(labels, n_classes):
        assert n_classes in [2, 3, 5, 10]
        bin_width = 3 // n_classes
        return (labels / bin_width).clamp(max=n_classes - 1)

    def corrupt(labels, n_classes, prob):
        is_corrupt = torch_bernoulli(prob, len(labels)).bool()
        return torch.where(is_corrupt, (labels + 1) % n_classes, labels)

    # Assign a label based on the digit
    labels = collapse_labels(labels, n_classes).float()

    # *Corrupt* label with probability 0.25 (default)
    labels = corrupt(labels, n_classes, label_prob)
    
    # Assign a color based on the label; flip the color with probability e
    colors = corrupt(labels, n_classes, color_prob)
    
    for i in range(len(images)):
        img = images[i]
        lbl = labels[i]
        color = colors[i]
        
        save_path = os.path.join(save_folder, str(int(lbl.numpy())), f'{i}_{int(color.numpy())}.png')
        save_torch_img(img, save_path)
        
        
n_classes = 3
label_flip = 0.25
save_environment_multiclass(os.path.join(datapath, 'train1'), train_img, train_lbl, 0.2, label_flip, n_classes=n_classes)
save_environment_multiclass(os.path.join(datapath, 'train2'), train_img, train_lbl, 0.1, label_flip, n_classes=n_classes)
save_environment_multiclass(os.path.join(datapath, 'val'),     val_img,   val_lbl, 0.2, label_flip, n_classes=n_classes)
save_environment_multiclass(os.path.join(datapath, 'test'),   test_img,  test_lbl, 0.9, label_flip, n_classes=n_classes)



import SimpleITK as sitk
import os, glob
import json
import numpy as np

### Select pre-defined/learned templates ### 
class_templates = {
    # '0':'...', 
    # '1':'...', 
    # '2':'...',
}

template_path = os.path.join(datapath, 'train1')


for ct in class_templates:
    if not os.path.exists(os.path.join(template_path, class_templates[ct])):
        print(f'template for {ct} does not exists')
        

for keyword in ['train1', 'train2', 'test', 'val']:
    dictout = {keyword:[]}
    
    for c in class_templates.keys():
        class_path = os.path.join(datapath, keyword, c)

        for fname in os.listdir(class_path):
            if not fname.endswith('.png'):
                continue
            smalldict = {}
            smalldict['source_img'] = os.path.join(template_path, class_templates[c])
            smalldict['target_img'] = os.path.join(class_path, fname)
            dictout[keyword].append(smalldict)


    savefilename = os.path.join(datapath, f'quickdraw_{keyword}.json')
    with open(savefilename, 'w') as fp:
        json.dump(dictout, fp)
