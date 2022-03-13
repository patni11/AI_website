import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
import pylab
import numpy as np
import cv2
from PIL import Image
import png
# load the models
from gan import Discriminator, Generator

D = Discriminator()
G = Generator()

# load weights
D.load_state_dict(torch.load('weights/weight_D.pth',map_location='cpu'))
G.load_state_dict(torch.load('weights/weight_G.pth',map_location='cpu'))

batch_size = 25
latent_size = 100

fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
fake_images = G(fixed_noise)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 3, 32, 32)
fake_images_np = fake_images_np.transpose((0, 2, 3, 1))
R, C = 5, 5
images = []
for i in range(batch_size): 
	images.append(fake_images_np[i])

vertical = np.vstack(tuple(images))
vertical *= 255.0/vertical.max()
print(np.amin(fake_images_np[1], axis=0))
print(np.amax(fake_images_np[1],axis = 0))
#matplotlib.image.imsave('name.png', vertical)
cv2.imwrite("filename.png", vertical)