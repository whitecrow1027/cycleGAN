import os
import torch
import torchvision.transforms as transforms
import visdom
import numpy as np
import time
import random
from PIL import Image


def vis_img(img,imageName,vis):
    #img=transforms.ToTensor()(img)
    vis.images(
        img,
        win=imageName,
        opts={'title':imageName}
    )

def vis_loss(lossTensor,lossName,count,vis):
    vis.line(
        Y=lossTensor,
        X=torch.as_tensor([count]),
        win=lossName,
        update='append',
        opts={'title':lossName}
    )

def vis_log(log,vis):
    vis.text(
        text=log,
        win='log',
        opts={'title':'log'}
    )


# dataroot = './result/maps/A/'
# vis=visdom.Visdom()

# for epoch in range(10):
#     imgName = dataroot + '%s.jpg' %(epoch+1)
#     img=Image.open(imgName)
# #    imgTensor=transforms.ToTensor()(img)
#     loss=torch.randn(1)
#     log="img: %s <br> loss: %s" %(imgName,loss.item())

#     vis_img(img,"img0",vis)
#     vis_loss(loss,'loss_A',epoch,vis)
#     vis_log(log,vis)
#     time.sleep(1)






#vis.image(imgTensor)

# for i in range(100):
#     y=random.randint(-10,10)
#     vis.line(
#         Y=np.array([y]),
#         X=np.array([i]),
#         win='loss',
#         update='append'
#     )
#     time.sleep(0.1)

# for i in range(100):
#     y=torch.randn(1)
#     vis.line(
#         Y=y,
#         X=torch.as_tensor([i]),
#         win='loss',
#         update='append',
#         opts={'title': 'loss'}
#     )
#     vis.text(text='loss1: %s <br> loss2' %(y.item()),win='log')
#     time.sleep(0.1)


    
