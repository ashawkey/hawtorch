import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import hawtorch
import hawtorch.io as io
from hawtorch import Trainer

import models 

from torchvision import datasets, transforms

args = io.load_json("mnist_config.json")
logger = io.logger(args["workspace_path"])

def create_trainer():
    print("Create Trainer")

    device = args["device"]

    model = getattr(models, args["model"])()

    objective = getattr(nn, args["objective"])()

    optimizer = getattr(optim, args["optimizer"])(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    lr_decay = lr_scheduler.StepLR(optimizer, step_size=args["lr_decay_step"], gamma=args["lr_decay"])

    metrics = [hawtorch.metrics.ClassificationAverager(10), ]

    loaders = create_loaders()

    trainer = Trainer(model, optimizer, lr_decay, objective, device, loaders, 
                  metrics=metrics, 
                  logger=logger,
                  workspace_path=args["workspace_path"],
                  eval_set="test",
                  )
    
    return trainer

def create_loaders():
    train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
                batch_size=args["train_batch_size"], 
                shuffle=True,
                num_workers=1,
                pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
                batch_size=args["test_batch_size"],
                shuffle=True,
                num_workers=1,
                pin_memory=True)

    print("create loaders", len(train_loader), len(test_loader))

    loaders = {
        "train": train_loader,
        "test": test_loader
    }

    return loaders



if __name__ == "__main__":
    trainer = create_trainer()
    trainer.train(args["epochs"])
    #trainer.evaluate()
    
