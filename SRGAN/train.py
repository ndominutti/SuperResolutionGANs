from tqdm import tqdm
import torch.optim as optim

def train_step(dataloader,
               generator,
               discriminator,
               vggloss,
               lr,
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


        ### GENERATOR LOSS
        



        
        