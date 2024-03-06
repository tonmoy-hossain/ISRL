import numpy as np
from numpy import zeros, newaxis
from numpy.random import seed
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from torchvision import datasets, models, transforms


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
    

class FCNModel(nn.Module):
    def __init__(self, channels, num_classes, B=8, drop_rate=0.25, flattened_dim = 798720):
        super(FCNModel, self).__init__()
        
#         dense_dim = B*16
        flattened_dim = 32768
        
#         self.fc_latent = nn.Linear(798720, flattened_dim)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, dense_dim),
#             nn.LeakyReLU(),
#             nn.ReLU(),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_rate),
            nn.Linear(dense_dim, num_classes)
        )
        
    def _conv_layer_set(self, in_c, out_c, kernel_size, stride, padding=0):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = x
        out = self.classifier(out)
        
        return out
    



def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

    
class AlexNet_3D(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self,  num_classes=2, dims=(224, 224, 224)):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.encoder = nn.Sequential(
            nn.Conv3d(1, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool3d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv3d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool3d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv3d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv3d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv3d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        
        self.flatten = nn.Flatten()
        encoder_output_shape = get_output_shape(self.encoder, (1, 1, *dims))
        self.flattened_dim = np.prod(encoder_output_shape) # product
        
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # , inplace=True
            nn.Linear(in_features=self.flattened_dim, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5), # , inplace=True
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Conv3d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
#         nn.init.constant_(self.encoder[4].bias, 1)
#         nn.init.constant_(self.encoder[10].bias, 1)
#         nn.init.constant_(self.encoder[12].bias, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        
        return self.classifier(x)
    

    
class VGG11_3D(nn.Module):
    def __init__(self, num_classes=2, dims=(224, 224, 224)):
        super(VGG11_3D, self).__init__()

        # Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        encoder_output_shape = get_output_shape(self.encoder, (1, 1, *dims))
        self.flattened_dim = np.prod(encoder_output_shape) # product
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
# import importlib
# import resnet
# importlib.reload(resnet)
# from resnet import generate_resnet_model

    
def get_adcn_encoder(channels=1, feat_dim=1024, expansion=4, type_name='conv3x3x3', norm_type='Instance'):
    conv = nn.Sequential()

    conv.add_module('conv0_s1',nn.Conv3d(channels, 4*expansion, kernel_size=1))

    if norm_type == 'Instance':
        conv.add_module('lrn0_s1',nn.InstanceNorm3d(4*expansion))
    else:
        conv.add_module('lrn0_s1',nn.BatchNorm3d(4*expansion))
    conv.add_module('relu0_s1',nn.ReLU(inplace=True))
    conv.add_module('pool0_s1',nn.MaxPool3d(kernel_size=3, stride=2))

    conv.add_module('conv1_s1',nn.Conv3d(4*expansion, 32*expansion, kernel_size=3,padding=0, dilation=2))

    if norm_type == 'Instance':
        conv.add_module('lrn1_s1',nn.InstanceNorm3d(32*expansion))
    else:
        conv.add_module('lrn1_s1',nn.BatchNorm3d(32*expansion))
    conv.add_module('relu1_s1',nn.ReLU(inplace=True))
    conv.add_module('pool1_s1',nn.MaxPool3d(kernel_size=3, stride=2))

    conv.add_module('conv2_s1',nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))

    if norm_type == 'Instance':
        conv.add_module('lrn2_s1',nn.InstanceNorm3d(64*expansion))
    else:
        conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
    conv.add_module('relu2_s1',nn.ReLU(inplace=True))
    conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=3, stride=2))

    conv.add_module('conv3_s1',nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))

    if norm_type == 'Instance':
        conv.add_module('lrn3_s1',nn.InstanceNorm3d(64*expansion))
    else:
        conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
    conv.add_module('relu3_s1',nn.ReLU(inplace=True))
    conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))
    
    if norm_type == 'Instance':
        conv.add_module('lrn4_s1',nn.InstanceNorm3d(64*expansion))
    else:
        conv.add_module('lrn3_s1',nn.BatchNorm3d(64*expansion))
    conv.add_module('relu4_s1',nn.ReLU(inplace=True))
    conv.add_module('pool3_s1',nn.MaxPool3d(kernel_size=5, stride=2))
    
    return conv


def get_mnist_encoder(channels=3, feat_dim=1024, expansion=4, type_name='conv3x3x3', norm_type='Instance'):
    conv = nn.Sequential()

    conv.add_module('conv0_s1',nn.Conv2d(channels, 4*expansion, kernel_size=1))

    if norm_type == 'Instance':
        conv.add_module('lrn0_s1',nn.InstanceNorm2d(4*expansion))
    else:
        conv.add_module('lrn0_s1',nn.BatchNorm2d(4*expansion))
    conv.add_module('relu0_s1',nn.ReLU(inplace=True))
    conv.add_module('pool0_s1',nn.MaxPool2d(kernel_size=3, stride=2))

    conv.add_module('conv1_s1',nn.Conv2d(4*expansion, 32*expansion, kernel_size=3,padding=0, dilation=2))

    if norm_type == 'Instance':
        conv.add_module('lrn1_s1',nn.InstanceNorm2d(32*expansion))
    else:
        conv.add_module('lrn1_s1',nn.BatchNorm2d(32*expansion))
    conv.add_module('relu1_s1',nn.ReLU(inplace=True))
    conv.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))

    conv.add_module('conv2_s1',nn.Conv2d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))

    if norm_type == 'Instance':
        conv.add_module('lrn2_s1',nn.InstanceNorm2d(64*expansion))
    else:
        conv.add_module('lrn2_s1',nn.BatchNorm2d(64*expansion))
    conv.add_module('relu2_s1',nn.ReLU(inplace=True))
    conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))

    conv.add_module('conv3_s1',nn.Conv2d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))

    if norm_type == 'Instance':
        conv.add_module('lrn3_s1',nn.InstanceNorm2d(64*expansion))
    else:
        conv.add_module('lrn2_s1',nn.BatchNorm2d(64*expansion))
    conv.add_module('relu3_s1',nn.ReLU(inplace=True))
    conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=5, stride=2))
    
    if norm_type == 'Instance':
        conv.add_module('lrn4_s1',nn.InstanceNorm2d(64*expansion))
    else:
        conv.add_module('lrn3_s1',nn.BatchNorm2d(64*expansion))
    conv.add_module('relu4_s1',nn.ReLU(inplace=True))
    conv.add_module('pool3_s1',nn.MaxPool2d(kernel_size=5, stride=2))
    
    return conv


def get_adcn_classifier():
    fcn = nn.Sequential()
    
    fcn = nn.Sequential(
        nn.Dropout(p=0.5), # , inplace=True
        nn.Linear(in_features=self.flattened_dim, out_features=4096),
        nn.ReLU(),
        nn.Dropout(p=0.5), # , inplace=True
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=num_classes),
    )


    fc6.add_module('fc6_s1',nn.Linear(d_model,512))
    fc6.add_module('lrn0_s1',nn.LayerNorm(512))
    fc6.add_module('fc6_s3',nn.Linear(512, out_dim))
        
# https://github.com/NYUMedML/CNN_design_for_AD/blob/master/models/models.py


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

            
def initialize_encoder_classifier(model_name, num_classes, feature_extract=False, use_pretrained=False, is_2d=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    encoder = None
    classifier = None
    num_ftrs = None
#     input_size = 224
    
    if not is_2d:
        if model_name == "resnet":
            """ Resnet18
            """
            m = generate_resnet_model(model_depth=18, n_input_channels=1, n_classes=2)
            encoder = m.encoder 
            classifier = m.classifier

        elif model_name == "alexnet":
            """ Alexnet
            """
            m = AlexNet_3D()
            encoder = m.encoder 
            classifier = m.classifier


        elif model_name == "vggnet":
            """ VGG11_bn
            """
            m = VGG11_3D()
            encoder = m.encoder 
            classifier = m.classifier
            
        else:
            print("Invalid model name, exiting...")
            exit()
    else:
        
        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

            encoder = nn.Sequential(*(list(model_ft.children())[:9])) 
            classifier = nn.Sequential(*(list(model_ft.children())[9:])) 

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

            encoder = nn.Sequential(*(list(model_ft.children())[:2])) 
            classifier = nn.Sequential(*(list(model_ft.children())[2:])) 


        elif model_name == "vggnet":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

            encoder = nn.Sequential(*(list(model_ft.children())[:2])) 
            classifier = nn.Sequential(*(list(model_ft.children())[2:])) 

        else:
            print("Invalid model name, exiting...")
            exit()

    return encoder, classifier, num_ftrs

# model_ft, input_size, num_ftrs = initialize_encoder('resnet')
# print(model_ft, input_size, num_ftrs)


# Create CNN Model
class CNNModel(nn.Module):
    
    def __init__(self, channels, dims, num_classes, B=8, drop_rate=0, is_2d=False, encoder_type='custom'):
        super(CNNModel, self).__init__()
        
        self.channels = channels
        
        self.is_custom = encoder_type == 'custom'
#         if not self.is_custom and self.channels == 1: # IF SOMETHING BREAKS, UNCOMMENT
#             channels = 3
            
        self.init_encoder(channels, B, num_classes, encoder_type, is_2d)
        
        print('encoder_type=', encoder_type)
        
        print((channels, *dims))
        
#         print(self.encoder)
        
        if is_2d and encoder_type == 'custom':
            encoder_output_shape = get_output_shape(self.encoder, (channels, *dims))
        else:
            encoder_output_shape = get_output_shape(self.encoder, (1, channels, *dims))
        flattened_dim = np.prod(encoder_output_shape) # product
        
        print('flattened_dim=', flattened_dim)
        
        self.flatten = nn.Flatten()
        
        self.init_classifier(flattened_dim, B, num_classes, drop_rate, encoder_type)
    
    
    def init_encoder(self, channels, B, num_classes, encoder_type, is_2d):
        if is_2d:
            if encoder_type == 'custom':
                self.encoder = get_mnist_encoder(channels=channels)
            else:
                self.encoder, classifier, num_ftrs = initialize_encoder_classifier(encoder_type, num_classes=num_classes)
            
        else: # 3D
            if encoder_type == 'custom':
                self.encoder = get_adcn_encoder()
        #         B = B*channels
        #         encoder = nn.Sequential(
        #             self._conv_layer_set(channels, B, kernel_size=5, stride=1),
        #             self._conv_layer_set(B, B*2, kernel_size=5, stride=1),
        #             self._conv_layer_set(B*2, B*8, kernel_size=3, stride=1)
        # #             self._conv_layer_set(B*8, B*16, kernel_size=3, stride=1)
        #         )

        #         encoder = nn.Sequential(
        #             self._conv_layer_set(channels, B, kernel_size=5, stride=2, padding='valid'), # 
        #             self._conv_layer_set(B, B*2, kernel_size=5, stride=1, padding='valid'),
        #             self._conv_layer_set(B*2, B*16, kernel_size=1, stride=1, padding='valid')
        #         )

        #         encoder = nn.Sequential(
        #             self._conv_layer_set(channels, B, kernel_size=5, stride=1),
        #             self._conv_layer_set(B, B*2, kernel_size=5, stride=1),
        #             self._conv_layer_set(B*2, B*16, kernel_size=5, stride=1),
        #         )

        #         encoder = nn.Sequential(
        #             self._conv_layer_set(channels, B, kernel_size=5, stride=1),
        #             self._conv_layer_set(B, B*2, kernel_size=5, stride=1),
        #             self._conv_layer_set(B*2, B*4, kernel_size=3, stride=1),
        #             self._conv_layer_set(B*4, B*8, kernel_size=3, stride=1),
        #             self._conv_layer_set(B*8, B*16, kernel_size=3, stride=1)
        #         )
            else:
                self.encoder, classifier, self.num_ftrs = initialize_encoder_classifier(encoder_type, num_classes=num_classes, is_2d=False)
            
    
    def init_classifier(self, flattened_dim, B, num_classes, drop_rate, encoder_type):
    
#         if encoder_type == 'custom':
            #         dense_dim = B*12
            #         classifier = nn.Sequential(
            #             nn.Linear(flattened_dim, dense_dim),
            #             nn.LeakyReLU(),
            #             nn.BatchNorm1d(dense_dim),
            #             nn.Dropout(p=drop_rate),
            #             nn.Linear(dense_dim, num_classes)
            #         )

        dense_dim = 1024
        print('inside: ', flattened_dim)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, dense_dim),
#             nn.LeakyReLU(),
#             nn.Dropout(p=drop_rate),
            nn.Linear(dense_dim, dense_dim//2),
#             nn.LeakyReLU(),
            nn.LayerNorm(dense_dim//2),
            nn.Dropout(p=drop_rate),
            nn.Linear(dense_dim//2, num_classes)
        )
            
#         else: # not custom = 'alexnet', 'vggnet', 'resnet', 'densenet'
#             pass # already defined in init_encoder

    
    def _conv_layer_set(self, in_c, out_c, kernel_size, stride, padding=0):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2)),
        )
        return conv_layer
    
    def _conv_layer_set_2d(self, in_c, out_c, kernel_size, stride, padding=0):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
#         print(x.shape)
#         if not self.is_custom and self.channels == 1:
#             x = x.squeeze()
#             x = torch.stack([x, x, x], 1)
#         print(x[0].shape)
#         plt.imshow(x[0].permute(1,2,0).detach().cpu().numpy())
#         plt.axis('off')
#         plt.show()
#         print(x.shape)
        out = self.encoder(x)
        out = self.flatten(out)
        out = self.classifier(out)
        
        return out
    

class JointCNNModel(CNNModel):
    def __init__(self, channels, dims, num_classes, B=8, drop_rate=0, op='cat', reg_dim=1024, is_2d=False, encoder_type='custom'):
        super(JointCNNModel, self).__init__(channels=channels, dims=dims, num_classes=num_classes, B=B, drop_rate=drop_rate, is_2d=is_2d, encoder_type=encoder_type)
        
        self.init_encoder(channels, B, num_classes, encoder_type, is_2d)
        
        print('encoder_type=', encoder_type)
        
        if is_2d and encoder_type == 'custom':
            encoder_output_shape = get_output_shape(self.encoder, (channels, *dims))
        else:
            encoder_output_shape = get_output_shape(self.encoder, (1, channels, *dims))        
        intensity_dim = np.prod(encoder_output_shape) # product
        
        print('flattened intensity dim=',intensity_dim)
        print('expected reg dim=', reg_dim)
        
#         reg_dim = 100352
        self.op = op
        if self.op == 'add':
            flattened_dim = intensity_dim
        elif self.op == 'cat':
            flattened_dim = intensity_dim*2
            print('inside flattened dim:', flattened_dim)
        
        self.flatten = nn.Flatten()
        
        self.decrease_linear = nn.Linear(reg_dim, intensity_dim)
        
#         self.increase_linear = nn.Linear(20736, 41472)
        
        dense_dim = B*12
        self.init_classifier(flattened_dim, dense_dim, num_classes, drop_rate, encoder_type)
    
    def fuse_features(self, x1, x2):
        if self.op == 'add':
            return x1+x2 # same dims
        elif self.op == 'cat':
            return torch.cat((x1, x2), 1) 
        
    def forward(self, x):
#         print(x.shape)
        try:
            x1, x2 = x

            x1 = self.encoder(x1)
            x1 = self.flatten(x1)
#         print('before fuse', x1.shape)
        except ValueError as v:
            x1 = x
            x2 = None
        
        if x2 is not None:
            x2 = self.flatten(x2)    
            x2 = self.decrease_linear(x2)
#             print('before fuse', x2.shape)
            final_latent = self.fuse_features(x1, x2)
        else:
            final_latent = x1
        
#         if test is True:
#             final_latent = self.increase_linear(final_latent)
#         print('after fuse', final_latent.shape)
#         print('flattened dim', self.flattened_dim)
        out = self.classifier(final_latent)
        
        return out

# summary(JointCNNModel(channels=1, num_classes=2, reg_dim=1024).to(dev), [(1, xDim, yDim, zDim), (1,32, 13, 16, 15)])