from tqdm import tqdm
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from loss import VGGLoss
import torch.nn as nn
import argparse
import logging
import sys
import dataset_handler
import copy
from datasets import load_dataset
from tensorboardX import SummaryWriter
from typing import Union


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def gen_loss(
    high_res: torch.Tensor,
    low_res: torch.Tensor,
    generator: nn.Moduel,
    discriminator: nn.Moduel,
    vggloss: nn.Moduel,
    bce: nn.Moduel,
) -> Union[float, float, float]:
    """Calculates the content, adversarial and perceptual loss

    Args:
        high_res (torch.Tensor): HR image
        low_res (torch.Tensor): LR image
        generator (nn.Moduel): Generator model (G)
        discriminator (nn.Moduel): Discriminator model (D)
        vggloss (nn.Module): content loss object
        bce (nn.Module): cross entropy object

    Returns:
        Union[float, float, float]: content_loss, adversarial_loss, perceptual_loss (content_loss + adversarial_loss)
    """
    gen_img = generator(low_res)
    disc_gen = discriminator(gen_img)
    content_loss = 0.006 * vggloss(gen_img, high_res)
    disc_gen = discriminator(gen_img)
    adversarial_loss = 10e-3 * bce(disc_gen, torch.ones_like(disc_gen))
    return content_loss, adversarial_loss, content_loss + adversarial_loss


def train_step(
    dataloader_train: torch.utils.data.IterableDataset,
    dataloader_test: torch.utils.data.IterableDataset,
    generator: nn.Module,
    discriminator: nn.Module,
    vggloss: nn.Module,
    bce: nn.Module,
    optimizer_disc: torch.optim.Optimizer,
    optimizer_gen: torch.optim.Optimizer,
    DEVICE: str,
    writer: SummaryWriter,
    epoch: int,
):
    """Manage a single train step (one epoch).

    Args:
        dataloader_train (torch.utils.data.IterableDataset): dataloader containing the training data
        dataloader_test (torch.utils.data.IterableDataset): dataloader containing the test data
        generator (nn.Module): Generator model (G)
        discriminator (nn.Module): Discriminator model (D)
        vggloss (nn.Module): content loss object
        bce (nn.Module): cross entropy object
        optimizer_disc (torch.optim.Optimizer): Discriminator optimizer
        optimizer_gen (torch.optim.Optimizer): Generator optimizer
        DEVICE (str): 'cpu' or 'cuda'
        writer (SummaryWriter): tensorboard SummaryWriter
        epoch (int): the current epoch number
    """
    global gstep
    global minimum_loss
    global best_generator
    global non_improving_rounds
    for datadict in tqdm(dataloader_train):
        high_res = datadict["hr_image"].to(DEVICE)
        low_res = datadict["lr_image"].to(DEVICE)
        gen_img = generator(low_res)

        ### DISCRIMINATOR LOSS
        disc_gen = discriminator(gen_img.detach())
        disc_real = discriminator(high_res)
        # Use BCEWithLogitLoss to train.
        # For real cases the calculus is between the output from de Discriminator for
        # the real images and a tensor full of 1s (because real images label is 1)
        # Then add some noise just to improve training stability and prevent the discriminator
        # from becoming too confident too early in training
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_gen = bce(disc_gen, torch.zeros_like(disc_gen))
        loss_disc = disc_loss_gen + disc_loss_real

        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        ### GENERATOR LOSS (Perceptual loss)
        _, _, perceptual_loss = gen_loss(
            high_res, low_res, generator, discriminator, vggloss, bce
        )
        optimizer_gen.zero_grad()
        perceptual_loss.backward()
        optimizer_gen.step()
        writer.add_scalar("train_perceptual_loss", perceptual_loss, global_step=gstep)
        writer.add_scalar("dicriminator_loss", loss_disc, global_step=gstep)
        gstep += 1

    logger.info(f"Validation epoch {epoch}...\n")
    perceptual_loss = validate(
        dataloader_test, generator, discriminator, vggloss, bce, DEVICE, writer
    )
    if minimum_loss == None or minimum_loss > perceptual_loss:
        minimum_loss = perceptual_loss
        best_generator = copy.deepcopy(generator)
    else:
        non_improving_rounds += 1


def validate(
    test_dataloader: torch.utils.data.IterableDataset,
    generator: nn.Module,
    discriminator: nn.Module,
    vggloss: nn.Module,
    bce: nn.Module,
    DEVICE: str,
    writer: SummaryWriter,
) -> float:
    """Validation step. Calculates the loss for every record in a batch, then
    returns de mean perceptual loss.
    Writes to tensorboard.

    Args:
        test_dataloader (torch.utils.data.IterableDataset): _description_
        generator (nn.Module): Generator model (G)
        discriminator (nn.Module): Discriminator model (D)
        vggloss (nn.Module): content loss object
        bce (nn.Module): cross entropy object
        DEVICE (str): 'cpu' or 'cuda'
        writer (SummaryWriter): tensorboard SummaryWriter

    Returns:
        float: AVG perceptual loss through all the test batches
    """
    content_loss_sum = 0.0
    adversarial_loss_sum = 0.0
    perceptual_loss_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for datadict in test_dataloader:
            high_res = datadict["hr_image"].to(DEVICE)
            low_res = datadict["lr_image"].to(DEVICE)
            content_loss, adversarial_loss, perceptual_loss = gen_loss(
                high_res, low_res, generator, discriminator, vggloss, bce
            )
            content_loss_sum += content_loss
            adversarial_loss_sum += adversarial_loss
            perceptual_loss_sum += perceptual_loss
            num_batches += 1
    writer.add_scalar(
        "test_content_loss", content_loss_sum / num_batches, global_step=gstep
    )
    writer.add_scalar(
        "test_adversarial_loss", adversarial_loss / num_batches, global_step=gstep
    )
    writer.add_scalar(
        "test_perceptual_loss", perceptual_loss / num_batches, global_step=gstep
    )
    return perceptual_loss / num_batches


def train(args):
    """
    Training wrapper. In charge of:
    * Download dataset
    * Preprocess data
    * Prepare dataloaders
    * Run training
    * Write summaries
    * Save the best model
    """
    if args.device == "available":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    writer = SummaryWriter("logs", comment=args.model_name)
    logger.info("Loading dataset...\n")
    train_dataset = load_dataset(
        "satellite-image-deep-learning/SODA-A", split="train[:1000]"
    )
    test_dataset = load_dataset(
        "satellite-image-deep-learning/SODA-A", split="train[1000:1500]"
    )
    logger.info("Creating dataloader...\n")
    custom_dataset_train = dataset_handler.CustomImageDataset(train_dataset)
    dataloader_train = DataLoader(
        custom_dataset_train, batch_size=args.train_batch_size, shuffle=True
    )
    custom_dataset_test = dataset_handler.CustomImageDataset(test_dataset)
    dataloader_test = DataLoader(
        custom_dataset_test, batch_size=args.test_batch_size, shuffle=True
    )
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vggloss = VGGLoss(device)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    optimizer_disc = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999)
    )
    optimizer_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    logger.info("Training started...\n")
    global gstep
    gstep = 0
    global minimum_loss
    minimum_loss = None
    global best_generator
    global non_improving_rounds
    non_improving_rounds = 0
    for epoch in range(args.epochs):
        train_step(
            dataloader_train,
            dataloader_test,
            generator,
            discriminator,
            vggloss,
            bce,
            optimizer_disc,
            optimizer_gen,
            device,
            writer,
            epoch,
        )
        if non_improving_rounds >= args.early_stopping_rounds:
            print("*" * 50 + "\n EARL STOPPING" + "\n" + "*" * 50)
            break

    torch.save(best_generator, f"{args.save_path}/generator_{args.model_name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--early-stopping-rounds", type=int, default=3)
    parser.add_argument("--device", type=str, default="available")
    parser.add_argument("--train-data-dir", type=str, default="data/train/")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=10e-3)
    train(parser.parse_args())
