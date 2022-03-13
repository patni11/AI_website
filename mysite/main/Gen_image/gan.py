import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils

class Generator(nn.Module):
    def __init__(self, out=3, inp=100, batch_size=64):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     inp, batch_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(batch_size * 8),
            nn.ReLU(True),
            # state size. (batch_size*8) x 4 x 4
            nn.ConvTranspose2d(batch_size * 8, batch_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(batch_size * 4),
            nn.ReLU(True),
            # state size. (batch_size*4) x 8 x 8
            nn.ConvTranspose2d(batch_size * 4, batch_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(batch_size * 2),
            nn.ReLU(True),
            # state size. (batch_size*2) x 16 x 16
            nn.ConvTranspose2d(batch_size * 2,     batch_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(    batch_size,      out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, out=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is (out) x 64 x 64
            nn.Conv2d(out, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    
