#test cycleGAN
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary

from network import Generator
from network import Discriminator
from network import weights_init
from datasets import CreatDatasets
from utils import tensor2img




parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str ,default='./datasets/maps',help='dataset root path')
parser.add_argument('--num_workers',type=int ,default=0,help='Number of workers for dataloader')
parser.add_argument('--batchsize',type=int ,default=1,help='batch size during training')
parser.add_argument('--input_nc',type=int ,default=3,help='channels number of input data')
parser.add_argument('--output_nc',type=int ,default=3,help='channels number of output data')
parser.add_argument('--image_size',type=int ,default=256,help='resize input image size')
parser.add_argument('--ngf',type=int ,default=64,help='Size of feature maps in generator')
parser.add_argument('--ndf',type=int ,default=64,help='Size of feature maps in discriminator')
parser.add_argument('--lr',type=float ,default=0.0002,help='Learning rate for optimizers')
parser.add_argument('--beta1',type=float ,default=0.5,help='Beta1 hyperparam for Adam optimizers')
parser.add_argument('--lambda1',type=float ,default=10.0,help='lamba of cycle loss')
parser.add_argument('--model_name',type=str,default='maps',help='model name')

opt = parser.parse_args()
print(opt)


#creat datasets
dataset = CreatDatasets(opt.dataroot,
                        transform=transforms.Compose([
                                transforms.Resize(opt.image_size),
                                transforms.CenterCrop(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]),
                        mode='test')
dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchsize,shuffle=True,num_workers=opt.num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)   

result_model_path="./result/model/%s" %(opt.model_name)

#creat and load network
netG_A2B=Generator(opt.input_nc,opt.output_nc,opt.ngf).to(device)
netG_B2A=Generator(opt.input_nc,opt.output_nc,opt.ngf).to(device)

netG_A2B.load_state_dict(torch.load('%s/%s_netG_A2B_final.pth' % (result_model_path,opt.model_name)))
netG_B2A.load_state_dict(torch.load('%s/%s_netG_B2A_final.pth' % (result_model_path,opt.model_name)))
# netG_A2B.load_state_dict(torch.load('result/model/netG_A2B.pth',map_location='cpu'))
# netG_B2A.load_state_dict(torch.load('result/model/netG_B2A.pth',map_location='cpu'))

# netG_A2B=torch.load('result/model/netG_A2B.pth')
# netG_B2A=torch.load('result/model/netG_B2A.pth')


netG_A2B.eval()
netG_B2A.eval()

#testing
for i,img in enumerate(dataloader,0):
    test_A = img['A'].to(device)
    test_B = img['B'].to(device)

    fake_A = netG_B2A(test_B)
    fake_B = netG_A2B(test_A)

    fake_A = 0.5*(fake_A.data + 1)
    fake_B = 0.5*(fake_B.data + 1)

    fake_A_img = tensor2img(fake_A[0])
    fake_B_img = tensor2img(fake_B[0])

    fake_A_img.save('./result/%s/A/%s.jpg' % (opt.model_name,i))
    fake_B_img.save('./result/%s/B/%s.jpg' % (opt.model_name,i))
    print('save img ',i)
    



