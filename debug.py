import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
from torchsummary import summary

from network import Generator
from network import Discriminator
from network import weights_init
from network import updateLr
from datasets import CreatDatasets
from utils import ImgBuffer


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str ,default='E:/lab/data/cycleGAN_dataset/maps',help='dataset root path')
parser.add_argument('--num_workers',type=int ,default=0,help='Number of workers for dataloader')
parser.add_argument('--batchsize',type=int ,default=1,help='batch size during training')
parser.add_argument('--input_nc',type=int ,default=3,help='channels number of input data')
parser.add_argument('--output_nc',type=int ,default=3,help='channels number of output data')
parser.add_argument('--image_size',type=int ,default=256,help='resize input image size')
parser.add_argument('--ngf',type=int ,default=64,help='Size of feature maps in generator')
parser.add_argument('--ndf',type=int ,default=64,help='Size of feature maps in discriminator')
parser.add_argument('--num_epochs',type=int ,default=10,help='number of epochs of training ')
parser.add_argument('--decay_epoch',type=int ,default=5,help='epoch that start to decaying learning rate ')
parser.add_argument('--lr',type=float ,default=0.0002,help='Learning rate for optimizers')
parser.add_argument('--beta1',type=float ,default=0.5,help='Beta1 hyperparam for Adam optimizers')
parser.add_argument('--lambda1',type=float ,default=10.0,help='lamba of cycle loss')

opt = parser.parse_args()
print(opt)

#creat datasets
dataset = CreatDatasets(opt.dataroot,
                        transform=transforms.Compose([
                                transforms.Resize(opt.image_size),
                                transforms.CenterCrop(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]),
                        mode='train')
dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchsize,shuffle=True,num_workers=opt.num_workers)

# print(dataloader.__len__())

# img=dataset.__getitem__(10)
# print(img['A'].size())

netG_A2B=Generator(opt.input_nc,opt.output_nc,opt.ngf)
netG_B2A=Generator(opt.input_nc,opt.output_nc,opt.ngf)
netD_A=Discriminator(opt.input_nc,opt.ndf)
netD_B=Discriminator(opt.input_nc,opt.ndf)

summary(netG_A2B,input_size=(3,256,256),device='cpu')
summary(netD_A,input_size=(3,256,256),device='cpu')

#instialize weights
netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

#print(netG_A2B)

#print(netD_A)

# Lossess
gan_loss = torch.nn.MSELoss()
cycle_consistency_loss = torch.nn.L1Loss()

#optimizers 
optimizerG_A2B = optim.Adam(netG_A2B.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerG_B2A = optim.Adam(netG_B2A.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerD_A = optim.Adam(netD_A.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerD_B = optim.Adam(netD_B.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))

lr_scheduler_G_A2B=optim.lr_scheduler.LambdaLR(optimizerG_A2B,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)
lr_scheduler_G_B2A=optim.lr_scheduler.LambdaLR(optimizerG_B2A,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)
lr_scheduler_D_A=optim.lr_scheduler.LambdaLR(optimizerD_A,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)
lr_scheduler_D_B=optim.lr_scheduler.LambdaLR(optimizerD_B,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)


real_label = torch.tensor(1.0)
fake_label = torch.tensor(0.0)

fake_buffer_A=ImgBuffer(50)
fake_buffer_B=ImgBuffer(50)

#print('label size: ',real_label.size())

for epoch in range(opt.num_epochs):
        for i,img in enumerate(dataloader,0):
                real_A = img['A']
                real_B = img['B']


                # img = fake_buffer.refresh(real_A)

                # print('img: ',img)
                # print('buffer len: ',len(fake_buffer.data))
                # pass

                # print('input size: ',real_A.size())
                # netG_A2B.zero_grad()

                # fake_B = netG_A2B(real_A)
                # print('output size: ',fake_B.size())

                # if_fake = netD_B(fake_B)
                # print('out size: ',if_fake.size())
                # #print('output: ',if_fake)
                # loss_G_A2B = gan_loss(if_fake,real_label)

                # print('loss_G_A2B: ',loss_G_A2B)
                print('i:',i,' epoch',epoch)
                netG_A2B.zero_grad()
                netG_B2A.zero_grad()

                #GAN loss
                fake_B = netG_A2B(real_A)
                fake_out = netD_B(fake_B)
                real_label=real_label.expand_as(fake_out)
                fake_label=fake_label.expand_as(fake_out)
                print('D out size: ',fake_out.size())
                print('label size: ',real_label.size())
                loss_G_A2B = gan_loss(fake_out,real_label)
                print(loss_G_A2B)

                fake_A = netG_B2A(real_B)
                fake_out = netD_A(fake_A)
                loss_G_B2A = gan_loss(fake_out,real_label)

                #cycle loss
                cycle_A = netG_B2A(fake_B)
                loss_cyc_A = cycle_consistency_loss(cycle_A,real_A)

                cycle_B = netG_A2B(fake_A)
                loss_cyc_B = cycle_consistency_loss(cycle_B,real_B)

                #full objective loss
                loss_G = loss_G_A2B + loss_G_B2A + opt.lambda1*(loss_cyc_A+loss_cyc_B)
                loss_G.backward()

                optimizerG_A2B.step()
                optimizerG_B2A.step()

                ###Discriminator###
                #DA
                netD_A.zero_grad()

                real_out = netD_A(real_A)
                loss_DA_real = gan_loss(real_out, real_label)
                fake_A = fake_buffer_A.refresh(fake_A)
                fake_out = netD_A(fake_A.detach())
                loss_DA_fake = gan_loss(fake_out,fake_label)
                loss_DA = (loss_DA_real+loss_DA_fake)*0.5
                loss_DA.backward()

                optimizerD_A.step()
                
                #DB
                netD_B.zero_grad()

                real_out = netD_B(real_B)
                loss_DB_real = gan_loss(real_out, real_label)
                fake_B = fake_buffer_B.refresh(fake_B)
                fake_out = netD_B(fake_B.detach())
                loss_DB_fake = gan_loss(fake_out,fake_label)
                loss_DB = (loss_DB_real+loss_DB_fake)*0.5
                loss_DB.backward()

                optimizerD_B.step()

                if i%100 == 0:
                        print( 'epoch: ',epoch,' step: ',i,' loss_G: ', loss_G ,' loss_DA: ', loss_DA, ' loss_DB: ',loss_DB)


        torch.save(netG_A2B,'result/model/netG_A2B.pth')
        torch.save(netG_B2A,'result/model/netG_B2A.pth')
        torch.save(netD_A,'result/model/netD_A.pth')
        torch.save(netD_B,'result/model/netD_B.pth')

                



    