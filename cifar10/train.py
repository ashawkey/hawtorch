import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import hawtorch
import hawtorch.io as io
from hawtorch import Trainer
from hawtorch.metrics import ClassificationMeter
from hawtorch.utils import backup

import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs.json')
parser = parser.parse_args()

config_file = parser.config

args = io.load_json(config_file)
logger = io.logger(args["workspace_path"])

names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def create_loaders():
    # transforms
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR10 dataset
    logger.info("Start creating datasets...")
    train_dataset = datasets.CIFAR10(root=args["data_path"], train=True, transform=transform_train, download=True)
    logger.info(f"Created train set! {len(train_dataset)}")
    test_dataset = datasets.CIFAR10(root=args["data_path"], train=False, transform=transform_test, download=True)
    logger.info(f"Created test set! {len(test_dataset)}")

    # Data Loader
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args["train_batch_size"],
                                shuffle=True,
                                pin_memory=True,
                                )

    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args["test_batch_size"],
                                shuffle=False,
                                pin_memory=True,
                                )

    return {"train":train_loader,
            "test":test_loader}

def create_trainer():
    logger.info("Start creating trainer...")
    device = args["device"]
    
    model = getattr(models, args["model"])()

    objective = getattr(nn, args["objective"])()
    optimizer = getattr(optim, args["optimizer"])(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args["lr_decay_step"], gamma=args["lr_decay"])
    metrics = [ClassificationMeter(10, names=names), ]
    loaders = create_loaders()

    trainer = Trainer(args, model, optimizer, scheduler, objective, device, loaders, logger,
                    metrics=metrics, 
                    workspace_path=args["workspace_path"],
                    eval_set="test",
                    input_shape=(3,32,32),
                    report_step_interval=-1,
                    )

    logger.info("Trainer Created!")

    return trainer

if __name__ == "__main__":
    backup(args["workspace_path"])
    trainer = create_trainer()
    trainer.train(args["epochs"])
    trainer.evaluate()
