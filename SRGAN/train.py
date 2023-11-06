from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from loss import VGGLoss
import torch.nn as nn
import argparse
import logging
import sys
import dataset_handler


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train_step(dataloader,
               generator,
               discriminator,
               vggloss,
               bce,
               optimizer_disc,
               optimizer_gen
              ):
    
    for high_res, low_res in tqdm(dataloader):
        high_res.to('cpu')
        low_res.to('cpu')
        gen_img   = generator(low_res)
        
        ### DISCRIMINATOR LOSS
        disc_gen  = discriminator(gen_img.detach())
        disc_real = discriminator(high_res)
        #Use BCEWithLogitLoss to train. 
        #For real cases the calculus is between the output from de Discriminator for
        #the real images and a tensor full of 1s (because real images label is 1)
        #Then add some noise just to improve training stability and prevent the discriminator
        #from becoming too confident too early in training
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_gen = bce(disc_gen, torch.zeros_like(disc_gen))
        loss_disc = disc_loss_fake + disc_loss_real

        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()


        ### GENERATOR LOSS (Perceptual loss)
        content_loss     = 0.006 * vggloss(gen_img, high_res) #0.006 equals to the 1/12.75 rescaling factor used in the paper
        disc_gen  = discriminator(gen_img)
        adversarial_loss = -1e-2*bce(disc_gen, torch.ones_like(disc_fake))
        perceptual_loss  = content_loss + adversarial_loss

        optimizer_gen.zero_grad()
        perceptual_loss.backward()
        optimizer_gen.step()


def train(args):
    """
    Main training pipeline
    """
    logger.info("Loading dataset...\n")
    train_dataset = dataset_handler.load_train_dataset(
        args.train_data_dir
    )        
    logger.info("Creating dataloader...\n")
    dataloader     = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    generator      = Generator()
    discriminator  = Discriminator()
    vggloss        = VGGLoss()
    bce            = nn.BCEWithLogitsLoss()
    mse            = nn.MSELoss()
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    logger.info("Training started...\n")
    for epoch in range(args.epochs):
        train_step(dataloader,
                   generator,
                   discriminator,
                   vggloss,
                   bce,
                   optimizer_disc,
                   optimizer_gen
                  )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-data-dir", type=str, default='data/train/')
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=10e-3)
    train(parser.parse_args())
    
    

    
        
        



        
        