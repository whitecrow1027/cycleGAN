import random
import os
import torch
import torchvision.transforms as transforms
from PIL import Image



def tensor2img(img_tensor):
    image = img_tensor.clone().cpu()
    image = image.squeeze()
    image = 0.5*(image.data+1)
    #print("image: type %s size %s" %(image.type(),image.size()))
    image = transforms.ToPILImage()(image)
    return image


def mkdir(path):
    if os.path.exists(path):
        print("path %s existed!" %(path))
    else:
        os.makedirs(path)
        print("make path %s " %(path)) 
    


class ImgBuffer():

    def __init__(self,max_len):
        self.max_len = max_len
        self.data = []
    
    # def refresh(self,data):
    #     self.data.append(data)
    #     if len(self.data) > self.max_len:
    #         del self.data[0]
    #     return self.data[random.randint(0,len(self.data)-1)]
        

    def refresh(self,data):
        if len(self.data) < self.max_len:
            self.data.append(data)
            return data
        else:
            p = random.uniform(0, 1)
            if p > 0.5:
                self.data.append(data)
                del self.data[0]
                return self.data[random.randint(0,len(self.data)-1)]
            else:
                return data



        
    
