# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageFile, ImageFilter
from torch.autograd import Variable

ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy
import scipy.io
import pdb
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

plt.ion()  # interactive mode

# for reproducibility
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['TORCH_HOME'] = 'D:/workplace/pytorchModel'
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


######################################################################
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ToPILImage(),
    #    AddSaltPepperNoise(0.05,1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# data_dir = 'D:/BaiduNetdiskDownload/office_caltech_10/office_caltech_10/amazon'

## split the data into train/validation/test subsets
# indices = list(range(len(image_dataset)))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.classifier = model._modules['avgpool']

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in model_ft._modules.items():
            if module_pos == 'layer4':
                for module_pos1, module1 in module._modules.items():
                    if module_pos1 == '1':
                        for module_pos2, module2 in module1._modules.items():
                            if module_pos2 == 'conv3':
                                x = module2(x)
                                x.register_hook(self.save_gradient)
                                conv_output = x
                            else:
                                x = module2(x)
                    else:
                        x = module1(x)
            else:
                x = module(x)

        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        #        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        output2 = model_output.data
        # Zero grads
        self.model.zero_grad()
        #        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=output2, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        beforeCam = cam
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam, weights, target, beforeCam


# extract one image
def singleWork(args):
    grad_cam = args[0]
    imgs = args[1]
    index = args[2]
    work_dir = args[3]
    savePath = work_dir + '/img' + str(index) + '.mat'
    if not os.path.exists(savePath):
        print('Deal with {} (save to {})...\n'.format(index, savePath))
        originImgs = recreate_image(imgs)
        originImgs = Image.fromarray(np.uint8(originImgs))
        imgs = Variable(imgs, requires_grad=True)
        cam, weights, target, beforeCam = grad_cam.generate_cam(imgs, target_class=None)
        scipy.io.savemat(savePath,
                         mdict={'weights': weights,
                                'target': target})
    else:
        print('Save {} fail, the file already exists'.format(index))


def weight_features(model, data_dir, work_dir, batchSize, workerSize):
    image_dataset = datasets.ImageFolder(data_dir, transform=data_transform)
    dataloader_for_feature_extraction = torch.utils.data.DataLoader(dataset=image_dataset, \
                                                                    shuffle=False, batch_size=batchSize, num_workers=0)
    #    model.eval()
    num_images = len(image_dataset)
    grad_cam = GradCam(model, target_layer=1)
    pool = ThreadPool(processes=workerSize)
    for index, (imgs, labels) in enumerate(dataloader_for_feature_extraction):
        print('Save heatmap weights for {} out of {} images'.format(index, num_images))
        tasks = []
        for k in range(len(labels)):
            img = imgs[k, :]
            img.unsqueeze_(0)
            tasks.append([grad_cam, img, index * batchSize + k + 1, work_dir])

        # append one
        pool.map(singleWork, tasks)
    #        if index >= 1:
    #            break

    return 0


from myResnet import resnet50_feature_extractor
import numpy as np
import torch
from misc_functions import get_example_params, recreate_image, save_class_activation_images
import matplotlib.pyplot as plt

model_ft = resnet50_feature_extractor(pretrained=True)
batchSize = 5
workerSize = 2
# 1
#     'F:/data/raw_Office31/Original_images/amazon',
#     'F:/data/raw_Office31/Original_images/dslr',
#     'F:/data/raw_Office31/Original_images/webcam',
#     'F:/data/raw_OfficeHome/OfficeHome/Art',
#     'F:/data/raw_OfficeHome/OfficeHome/Clipart',
#     'F:/data/raw_OfficeHome/OfficeHome/Product',
#     'F:/data/raw_OfficeHome/OfficeHome/RealWorld'
#     'F:/data/raw_Animals_with_Attributes2 (AWA2)/JPEGImages',
#     'F:/data/raw_imageCLEF/image_CLEF_raw/b',
#     'F:/data/raw_imageCLEF/image_CLEF_raw/c',
#     'F:/data/raw_imageCLEF/image_CLEF_raw/i',
#     'F:/data/raw_imageCLEF/image_CLEF_raw/p',
#     'F:/data/raw_Modern-Office-31/Modern-Office-31/amazon',
#     'F:/data/raw_Modern-Office-31/Modern-Office-31/dslr',
#     'F:/data/raw_Modern-Office-31/Modern-Office-31/synthetic',
#     'F:/data/raw_Modern-Office-31/Modern-Office-31/webcam',
data_dir_list = [
    'F:/data/raw_office_caltech_10/office_caltech_10/amazon',
    'F:/data/raw_office_caltech_10/office_caltech_10/caltech',
    'F:/data/raw_office_caltech_10/office_caltech_10/dslr',
    'F:/data/raw_office_caltech_10/office_caltech_10/webcam',
    'F:/data/raw_Adaptiope/Adaptiope/synthetic',
    'F:/data/raw_Adaptiope/Adaptiope/real_life',
    'F:/data/raw_Adaptiope/Adaptiope/product_images',
]
#
#     'E:/data/params/Office31/amazon',
#     'E:/data/params/Office31/dslr',
#     'E:/data/params/Office31/webcam',
#     'E:/data/params/OfficeHome/Art',
#     'E:/data/params/OfficeHome/Clipart',
#     'E:/data/params/OfficeHome/Product',
#     'E:/data/params/OfficeHome/RealWorld'
#     'E:/data/params/AWA2',
#     'E:/data/params/ImageCLEF/b',
#     'E:/data/params/ImageCLEF/c',
#     'E:/data/params/ImageCLEF/i',
#     'E:/data/params/ImageCLEF/p',
#     'E:/data/params/Modern-Office-31/amazon',
#     'E:/data/params/Modern-Office-31/dslr',
#     'E:/data/params/Modern-Office-31/synthetic',
#     'E:/data/params/Modern-Office-31/webcam',

work_dir_list=[
    'E:/data/params/Office10/amazon',
    'E:/data/params/Office10/caltech',
    'E:/data/params/Office10/dslr',
    'E:/data/params/Office10/webcam',
    'E:/data/params/Adaptiope/product_images',
    'E:/data/params/Adaptiope/real_life',
    'E:/data/params/Adaptiope/synthetic',
]


for index in range(len(data_dir_list)):
    weight_features(model_ft, data_dir=data_dir_list[index],
                    work_dir=work_dir_list[index],
                    batchSize=batchSize, workerSize=workerSize)

