from tqdm import tqdm
import torch.optim as optim

def train_step(dataloader,
               generator,
               discriminator,
               vggloss,
               lr
              ):

    optimizer = optim.Adam(generator.parameters(), lr=lr)
    
    for high_res, low_res in tqdm(dataloader):
        high_res.to('cpu')
        low_res.to('cpu')

        #Use Mixed Precision Training
        with torch.cpu.amp.autocast():
            gen_img   = generator(low_res)
            disc_gen_img  = discriminator(gen_img.detach())
            disc_real_img = discriminator(high_res)
            disc_loss = (
                -(torch.mean(disc_real_img) - torch.mean(disc_gen_img))
            )

        
        