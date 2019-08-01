#train cycleGAN
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import visdom
from torchsummary import summary

from network import Generator
from network import Discriminator
from network import weights_init
from network import updateLr
from datasets import CreatDatasets
from utils import ImgBuffer
from utils import tensor2img
from utils import mkdir
from vis import vis_img
from vis import vis_log
from vis import vis_loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str ,default='./datasets/maps',help='dataset root path')
parser.add_argument('--model_name',type=str,default='maps',help='model name')
parser.add_argument('--num_workers',type=int ,default=0,help='Number of workers for dataloader')
parser.add_argument('--batchsize',type=int ,default=1,help='batch size during training')
parser.add_argument('--input_nc',type=int ,default=3,help='channels number of input data')
parser.add_argument('--output_nc',type=int ,default=3,help='channels number of output data')
parser.add_argument('--image_size',type=int ,default=256,help='resize input image size')
parser.add_argument('--ngf',type=int ,default=64,help='Size of feature maps in generator')
parser.add_argument('--ndf',type=int ,default=64,help='Size of feature maps in discriminator')
parser.add_argument('--num_epochs',type=int ,default=200,help='number of epochs of training ')
parser.add_argument('--decay_epoch',type=int ,default=100,help='epoch that start to decaying learning rate ')
parser.add_argument('--lr',type=float ,default=0.0002,help='Learning rate for optimizers')
parser.add_argument('--beta1',type=float ,default=0.5,help='Beta1 hyperparam for Adam optimizers')
parser.add_argument('--lambda1',type=float ,default=10.0,help='lamba of cycle loss')
parser.add_argument('--lambda_idt',type=float,default=0.5,help='use identity mapping')
parser.add_argument('--visualize',action="store_true",help='if use visualize')
parser.add_argument('--log_freq',type=int,default=10,help='log frequency')
parser.add_argument('--start_epoch',type=int,default=0,help="start epoch number,must be divied by 10")

opt = parser.parse_args()
print('training option',opt)

#creat datasets
dataset = CreatDatasets(opt.dataroot,
                        transform=transforms.Compose([
                                transforms.Resize(opt.image_size),
                                transforms.CenterCrop(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]),
                        mode='train')
dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchsize,shuffle=True,num_workers=opt.num_workers,drop_last=True)

#make reference dir
result_model_path="./result/model/%s" %(opt.model_name)
log_path="./log/%s" %(opt.model_name)
mkdir(result_model_path)
mkdir(log_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device: ',device)   
#creat network
netG_A2B=Generator(opt.input_nc,opt.output_nc,opt.ngf).to(device)
netG_B2A=Generator(opt.input_nc,opt.output_nc,opt.ngf).to(device)
netD_A=Discriminator(opt.input_nc,opt.ndf).to(device)
netD_B=Discriminator(opt.input_nc,opt.ndf).to(device)

#instialize weights
if opt.start_epoch == 0:
        netG_A2B.apply(weights_init)
        netG_B2A.apply(weights_init)
        netD_A.apply(weights_init)
        netD_B.apply(weights_init)
        print("training start from begining")
else:   #read trained network param
        netG_A2B.load_state_dict(torch.load('%s/%s_netG_A2B_ep%s.pth' % (result_model_path,opt.model_name,opt.start_epoch)))
        netG_B2A.load_state_dict(torch.load('%s/%s_netG_B2A_ep%s.pth' % (result_model_path,opt.model_name,opt.start_epoch)))
        netD_A.load_state_dict(torch.load('%s/%s_netD_A_ep%s.pth' % (result_model_path,opt.model_name,opt.start_epoch)))
        netD_B.load_state_dict(torch.load('%s/%s_netD_B_ep%s.pth' % (result_model_path,opt.model_name,opt.start_epoch)))
        print("training start from epoch %s" %(opt.start_epoch))

#print(netG_A2B,netG_B2A,netD_A,netD_B)
summary(netG_A2B,input_size=(3,256,256))
summary(netG_B2A,input_size=(3,256,256))
summary(netD_A,input_size=(3,256,256))
summary(netD_B,input_size=(3,256,256))

# Lossess
gan_loss = torch.nn.MSELoss()    #LSGAN
cycle_consistency_loss = torch.nn.L1Loss()
if opt.lambda_idt > 0:
        idt_loss = torch.nn.L1Loss()

#optimizers 
optimizerG_A2B = optim.Adam(netG_A2B.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerG_B2A = optim.Adam(netG_B2A.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerD_A = optim.Adam(netD_A.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
optimizerD_B = optim.Adam(netD_B.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))

lr_scheduler_G_A2B=optim.lr_scheduler.LambdaLR(optimizerG_A2B,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)
lr_scheduler_G_B2A=optim.lr_scheduler.LambdaLR(optimizerG_B2A,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)
lr_scheduler_D_A=optim.lr_scheduler.LambdaLR(optimizerD_A,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)
lr_scheduler_D_B=optim.lr_scheduler.LambdaLR(optimizerD_B,lr_lambda=updateLr(opt.num_epochs,opt.decay_epoch).update)

real_label = torch.tensor(1.0).to(device)
fake_label = torch.tensor(0.0).to(device)

fake_buffer_A=ImgBuffer(50)
fake_buffer_B=ImgBuffer(50)

if opt.visualize:
        vis=visdom.Visdom(env='cycleGAN')
        vis_update=0



#calculating time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
#training
for epoch in range(opt.start_epoch,opt.num_epochs):
        for i,img in enumerate(dataloader,0):

                real_A = img['A'].to(device)
                real_B = img['B'].to(device)

                ###Generator###
                netG_A2B.zero_grad()
                netG_B2A.zero_grad()

                #GAN loss
                fake_B = netG_A2B(real_A)
                fake_out = netD_B(fake_B)
                real_label=real_label.expand_as(fake_out)
                fake_label=fake_label.expand_as(fake_out)
                loss_G_A2B = gan_loss(fake_out,real_label)

                fake_A = netG_B2A(real_B)
                fake_out = netD_A(fake_A)
                loss_G_B2A = gan_loss(fake_out,real_label)

                #cycle loss
                cycle_A = netG_B2A(fake_B)
                loss_cyc_A = cycle_consistency_loss(cycle_A,real_A)

                cycle_B = netG_A2B(fake_A)
                loss_cyc_B = cycle_consistency_loss(cycle_B,real_B)

                #identity loss
                if opt.lambda_idt > 0:
                        idt_A = netG_A2B(real_B)
                        loss_idt_A = idt_loss(idt_A,real_B)

                        idt_B = netG_B2A(real_A)
                        loss_idt_B = idt_loss(idt_B,real_A)
                else:
                        loss_idt_A=0
                        loss_idt_B=0

                #full objective loss
                loss_G = loss_G_A2B + loss_G_B2A + opt.lambda1*(loss_cyc_A+loss_cyc_B+opt.lambda_idt*(loss_idt_A+loss_idt_B))
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

                if i%opt.log_freq == 0:
                        #end calculating time
                        end.record()
                        # Waits for everything to finish running
                        torch.cuda.synchronize()

                        log = "time: %ss <br> epoch: %s  step: %s <br> loss_G: %s <br> loss_DA: %s <br> loss_DB: %s <br> loss_cyc_A: %s <br> loss_cyc_B: %s <br> loss_idt_A: %s <br> loss_idt_B: %s" \
                                %(start.elapsed_time(end)/1000,epoch,i,loss_G.item(),loss_DA.item(),loss_DB.item(),loss_cyc_A.item(),loss_cyc_B.item(),loss_idt_A.item(),loss_idt_B.item())
                        #print( 'epoch: ',epoch,' step: ',i,' loss_G: ', loss_G ,' loss_DA: ', loss_DA, ' loss_DB: ',loss_DB)
                        print(log)
                        if opt.visualize:
                                vis_log(log,vis)

                                vis_img(fake_A,"fake_A",vis)
                                vis_img(fake_B,"fake_B",vis)
                                vis_loss(loss_G.reshape(-1),"loss_G",vis_update,vis)
                                vis_loss(loss_DA.reshape(-1),"loss_DA",vis_update,vis)
                                vis_loss(loss_DB.reshape(-1),"loss_DB",vis_update,vis)
                                vis_loss(loss_cyc_A.reshape(-1),"loss_cyc_A",vis_update,vis)
                                vis_loss(loss_cyc_B.reshape(-1),"loss_cyc_B",vis_update,vis)
                                vis_update+=1
                        if i%1000 == 0:
                                #fake_A = 0.5*(fake_A.data + 1)
                                #fake_B = 0.5*(fake_B.data + 1)
                                fake_A_img = tensor2img(fake_A[0])
                                fake_B_img = tensor2img(fake_B[0])
                                cycle_A_img = tensor2img(cycle_A[0])
                                cycle_B_img = tensor2img(cycle_B[0])
                                idt_A_img = tensor2img(idt_A[0])
                                idt_B_img = tensor2img(idt_B[0])
                                real_A_img = tensor2img(real_A[0])
                                real_B_img = tensor2img(real_B[0])
                                fake_A_img.save('%s/%s_%s_A.jpg' % (log_path,epoch,i))
                                fake_B_img.save('%s/%s_%s_B.jpg' % (log_path,epoch,i))
                                cycle_A_img.save('%s/%s_%s_cycle_A.jpg' % (log_path,epoch,i))
                                cycle_B_img.save('%s/%s_%s_cycle_B.jpg' % (log_path,epoch,i))
                                idt_A_img.save('%s/%s_%s_idt_A.jpg' % (log_path,epoch,i))
                                idt_B_img.save('%s/%s_%s_idt_B.jpg' % (log_path,epoch,i))
                                real_A_img.save('%s/%s_%s_real_A.jpg' % (log_path,epoch,i))
                                real_B_img.save('%s/%s_%s_real_B.jpg' % (log_path,epoch,i))

        
                        start.record()

        lr_scheduler_G_A2B.step()
        lr_scheduler_G_B2A.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if epoch%10 == 0:       
                torch.save(netG_A2B.state_dict(),'%s/%s_netG_A2B_ep%s.pth' % (result_model_path,opt.model_name,epoch))
                torch.save(netG_B2A.state_dict(),'%s/%s_netG_B2A_ep%s.pth' % (result_model_path,opt.model_name,epoch))
                torch.save(netD_A.state_dict(),'%s/%s_netD_A_ep%s.pth' % (result_model_path,opt.model_name,epoch))
                torch.save(netD_B.state_dict(),'%s/%s_netD_B_ep%s.pth' % (result_model_path,opt.model_name,epoch))

torch.save(netG_A2B.state_dict(),'%s/%s_netG_A2B_final.pth' % (result_model_path,opt.model_name))
torch.save(netG_B2A.state_dict(),'%s/%s_netG_B2A_final.pth' % (result_model_path,opt.model_name))
torch.save(netD_A.state_dict(),'%s/%s_netD_A_final.pth' % (result_model_path,opt.model_name))
torch.save(netD_B.state_dict(),'%s/%s_netD_B_final.pth' % (result_model_path,opt.model_name))

        











 



