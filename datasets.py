import os
import random

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def Filelist(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, files, filenames in os.walk(dir):
        for filename in filenames:
                path = os.path.join(root, filename)
                images.append(path)
    return images[:]

class CreatDatasets(Dataset):
    """ creat an A to B datasets"""

    def __init__(self,dataroot,transform=None,mode='train'):
        self.root=dataroot
        self.transform=transform
        self.dir_A=os.path.join(self.root,mode+'A')
        self.dir_B=os.path.join(self.root,mode+'B')
        self.list_A=Filelist(self.dir_A)
        self.list_B=Filelist(self.dir_B)

    def __getitem__(self,index):
        img_A=self.transform( Image.open(self.list_A[index]) )
        img_B=self.transform( Image.open(self.list_B[ random.randint(0,len(self.list_B)-1) ]) )
        return {'A':img_A, 'B':img_B}

    def __len__(self):
        return max( len(self.list_A),len(self.list_B) )
