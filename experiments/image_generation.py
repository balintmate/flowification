import sys
sys.path.append('.')


import argparse
from experiments.architectures import getArchitecture
from models.transforms import CompositeTransform

from models.flows import Flow
from models.distributions.standard_normal import LogNormal
from dataloader import Data
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import wandb
import math
import os




parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--architecture", type=str)
parser.add_argument("--data", type=str, choices=["mnist", "cifar"])
args = parser.parse_args()

wandb_key = open('wandb.key', 'r').read()
if wandb_key:
    wandb.login(key=wandb_key)
    run = wandb.init(project="Flowification", tags=[args.data])
else:
    run = wandb.init(project="Flowification", mode='offline',tags=[args.data])


wandb.run.log_code(".")
wandb.config.update({"lr": args.lr, "epochs": args.epochs,
                    "architecture": args.architecture})
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trainloader, testloader = Data(args.data, batch_size=128)


dataiter = iter(trainloader)
X, _ = dataiter.next()
num_pixels = X.size(1)*X.size(2)*X.size(3)


class FlowifiedNet(Flow):
    def __init__(self, data, architecture):
        layers, latent_dim = getArchitecture(data, architecture)
        super(FlowifiedNet, self).__init__(
            CompositeTransform(layers), LogNormal(latent_dim))


model = FlowifiedNet(args.data, args.architecture)
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, args.epochs * len(trainloader))



wandb.config.update({"Mparams": f"{sum(p.numel() for p in model.parameters() if p.requires_grad) // 10 ** 5 / 10}M"})


for epoch in range(1, args.epochs + 1):
    for step, data in enumerate(trainloader, 0):
        mini_X, mini_Y = data
        mini_X = mini_X.to(device)
        z, logL = model.forward(mini_X)
        opt.zero_grad()
        (-logL.mean()).backward()
        opt.step()
        scheduler.step()
        wandb.log(
            {
                "log-likelihood": logL.mean().item(),
                "BPD train": -logL.mean().item() / math.log(2) / num_pixels,
                "epoch": epoch,
                "LR": scheduler.get_last_lr()[0]
            }
        )



    with torch.no_grad():
        L_test = 0
        for step, data in enumerate(testloader, 0):
            mini_X, mini_Y = data
            mini_X = mini_X.to(device)
            _, logL = model.forward(mini_X)
            L_test += logL.sum().item()
        wandb.log(
            {
                "BPD test": -L_test / len(testloader.dataset) / math.log(2) / num_pixels,
                "epoch": epoch,
            }
        )
        # Plotting to wandb
        NUM_SAMPLES = 15

        # Reconstruction
        dataiter = iter(trainloader)
        X, _ = dataiter.next()
        orig = X[:NUM_SAMPLES].clone().cpu()

        z = model.forward(X[:NUM_SAMPLES].to(device), flag="mean")[0]
        recon_mean = model.inverse(z, flag='mean').cpu().detach()
        recon_sample = model.inverse(z, flag='sample').cpu().detach()

        batch = torch.cat((orig, recon_mean, recon_sample), 0)
        batch = torch.nn.ReLU()(batch)  # why?
        batch = 1 - torch.nn.ReLU()(1 - batch)

        grid_img = make_grid(batch, nrow=NUM_SAMPLES)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis('off')
        wandb.log({"Reconstruction": wandb.Image(plt)})
        plt.close()

        # Generation
        z = torch.randn(
            10*NUM_SAMPLES, model.base_distribution.dimension)
        z = z.to(device)
        samples = model.sample(z=z, flag="sample").cpu().detach()
        samples = torch.nn.ReLU()(samples)
        samples = 1 - torch.nn.ReLU()(1 - samples)

        grid_img = make_grid(samples, nrow=NUM_SAMPLES)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis('off')
        wandb.log({"Generation (always sample)": wandb.Image(plt)})
        plt.close()

        samples = model.sample(z=z, flag="mean").cpu().detach()
        samples = torch.nn.ReLU()(samples)
        samples = 1 - torch.nn.ReLU()(1 - samples)

        grid_img = make_grid(samples, nrow=NUM_SAMPLES)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis('off')
        wandb.log({"Generation (always mean)": wandb.Image(plt)})
        plt.close()