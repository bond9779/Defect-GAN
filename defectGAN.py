import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

output_path = "Output/run1"
if not os.path.exists(output_path):
    os.makedirs(output_path)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
# criterion_GAN = torch.nn.BCELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G = GeneratorResNet(input_shape, opt.n_residual_blocks)
F = GeneratorResNet(input_shape, opt.n_residual_blocks)
D1g = Discriminator(input_shape)
D1b = Discriminator(input_shape)
D2g = Discriminator(input_shape)
D2b = Discriminator(input_shape)

if cuda:
    G = G.cuda()
    F = F.cuda()
    D1g = D1g.cuda()
    D1b = D1b.cuda()
    D2g = D2g.cuda()
    D2b = D2b.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

""" if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch))) """
# else:
    # Initialize weights
G.apply(weights_init_normal)
F.apply(weights_init_normal)
D1g.apply(weights_init_normal)
D1b.apply(weights_init_normal)
D2g.apply(weights_init_normal)
D2b.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    list(G.parameters()) + list(F.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

# optimizer_F = torch.optim.Adam(
#     F.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
# )


optimizer_DG = torch.optim.Adam(list(D1g.parameters()) + list(D2g.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_DB = torch.optim.Adam(list(D1b.parameters()) + list(D2b.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

""" optimizer_D2g = torch.optim.Adam(D2g.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2b = torch.optim.Adam(D2b.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) """

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

""" lr_scheduler_F = torch.optim.lr_scheduler.LambdaLR(
    optimizer_F, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
) """

lr_scheduler_DG = torch.optim.lr_scheduler.LambdaLR(
    optimizer_DG, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(
    optimizer_DB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
""" lr_scheduler_D2g = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2g, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2b = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2b, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
) """

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.Resize((128, 128)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("C:\\Users\\admin\\Documents\\MyDocs\\Datasets\\horse2zebra\\horse2zebra", transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("C:\\Users\\admin\\Documents\\MyDocs\\Datasets\\horse2zebra\\horse2zebra", transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)




def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    vldr = iter(val_dataloader)
    G.eval()
    # F.eval()
    """ real_ok = Variable(imgs["A"].type(Tensor)) #g
    fake_defect = G(real_ok) # G(g)
    real_defect = Variable(imgs["B"].type(Tensor)) #b
    fake_ok = F(real_defect) # F(b)
    # Arange images along x-axis
    real_A = make_grid(real_ok, nrow=5, normalize=True)
    real_B = make_grid(real_defect, nrow=5, normalize=True)
    fake_A = make_grid(fake_ok, nrow=5, normalize=True)
    fake_B = make_grid(fake_defect, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1) """
    if not os.path.exists(os.path.join(output_path, f"images", f"{batches_done}")):
        os.makedirs(os.path.join(output_path, f"images", f"{batches_done}"))
    for q in range(len(vldr)):
        imgs = next(vldr)
        real_ok = Variable(imgs["A"].type(Tensor)) #g
        fake_defect = G(real_ok) # G(g)
        save_image(fake_defect, os.path.join(output_path, f"images", f"{batches_done}", f"{q}.png"), normalize=True)

def disc_loss(real, generated):
    real_loss = criterion_GAN(real, torch.ones_like(real))
    generated_loss = criterion_GAN(generated, torch.zeros_like(generated))
    return (real_loss + generated_loss) * 0.5

# def gen_loss(generated):
#     return criterion_GAN(generated, torch.ones_like(generated))
# ----------
#  Training
# ----------
if __name__ == "__main__":
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            g = Variable(batch["A"].type(Tensor)) # g - Sample real defect free Image
            b = Variable(batch["B"].type(Tensor)) # b - Sample real defect image

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((g.size(0), *D1g.output_shape))), requires_grad=False) # Ones
            fake = Variable(Tensor(np.zeros((b.size(0), *D1b.output_shape))), requires_grad=False) # Zeros

            # Train discriminators D1g and D2g, D1b, D2b
            f_b = F(b) # Obtain fake defect free image g^
            g_g = G(g) # Obtain fake defect image b^

            d1g_real = D1g(g)
            d1g_fake = D2g(f_b.detach())
            d1g_loss = criterion_GAN(d1g_real, valid) + criterion_GAN(d1g_fake, fake) 

            d2g_real = D2g(g)
            d2g_fake = D2g(f_b.detach())
            d2g_loss = criterion_GAN(d2g_real, fake) + criterion_GAN(d2g_fake, valid) 

            DG_loss = (d1g_loss + d2g_loss)/2

            optimizer_DG.zero_grad()
            DG_loss.backward()
            optimizer_DG.step()

            d1b_real = D1b(b)
            d1b_fake = D1b(g_g.detach())
            d1b_loss = criterion_GAN(d1b_real, valid) + criterion_GAN(d1b_fake, fake) 

            d2b_real = D2b(b)
            d2b_fake = D2g(g_g.detach())
            d2b_loss = criterion_GAN(d2b_real, fake) + criterion_GAN(d2b_fake, valid) 

            DB_loss = (d1b_loss + d2b_loss)/2

            optimizer_DB.zero_grad()
            DB_loss.backward()
            optimizer_DB.step()

            # Train generators

            g_f_b = G(f_b)
            f_g_g = F(g_g)
            

            # ------------------
            #  Train Generators
            # ------------------
            
            # Identity loss
            loss_id_A = criterion_identity(G(g), g)
            loss_id_B = criterion_identity(F(b), b)

            loss_identity = (loss_id_A + loss_id_B) / 2
            
            optimizer_G.zero_grad()

            # GAN loss
            
            g_g = G(g) # Obtain fake defect image

            g_f_b = G(f_b) # Cycle f_b
            f_g_g = F(g_g) # Cycle g_g

            F_loss = criterion_GAN(D1g(f_b), valid) + criterion_GAN(D2g(f_b), fake)\
                + criterion_identity(g_f_b, b)
            
            G_loss = criterion_GAN(D1b(g_g), valid) + criterion_GAN(D2b(g_g), fake)\
                + criterion_identity(f_g_g, g)
            
            gen_loss = (F_loss + G_loss) / 2
            optimizer_G.zero_grad()
            gen_loss.backward()
            optimizer_G.step()
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [DG loss: %f] [DB loss: %f] [G loss: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    DG_loss.item(),
                    DB_loss.item(),
                    gen_loss.item(),
                    # loss_cycle.item(),
                    # loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_DG.step()
        lr_scheduler_DB.step()
        # lr_scheduler_D_B.step()

        """ if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch)) """
