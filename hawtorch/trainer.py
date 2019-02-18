import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import hawtorch
import hawtorch.nn as hnn
from hawtorch.io import logger
from hawtorch.utils import DelayedKeyboardInterrupt

class Trainer(object):
    """Base trainer class. 
    """

    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler, 
                 objective, 
                 device,
                 dataloaders,
                 metrics=[],
                 logger,
                 workspace_path=None, 
                 use_checkpoint=-1,
                 eval_set="val",
                 eval_interval=1,
                 save_interval=1,
                 report_step_interval=300,
                 ):
                 
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.objective = objective
        self.device = device
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.log = logger
        self.workspace_path = workspace_path
        self.use_checkpoint = use_checkpoint
        self.eval_set = eval_set
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.report_step_interval = report_step_interval

        self.model.to(self.device)


        if self.workspace_path is not None:
            os.makedirs(self.workspace_path, exist_ok=True)
            if self.use_checkpoint == 0:
                self.log.info("Train from zero")
            elif self.use_checkpoint < 0:
                self.log.info("Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint > 0:
                self.log.info(f"Loading checkpoint {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        self.log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        for metric in self.metrics:
            metric.clear()

        self.epoch = 1
        self.global_step = 1
        self.stats = {
            "StepLoss": [],
            "EpochLoss": [],
            }


    def train(self, max_epochs):
        for epoch in range(self.epoch, max_epochs+1):
            self.epoch = epoch
            self.train_one_epoch()

            if self.epoch % self.eval_interval == 0:
                self.evaluate()

            if self.workspace_path is not None and self.epoch % self.save_interval == 0:
                self.save_checkpoint()
                self.log.info(f"Saved checkpoint {epoch} successfully.")

        self.log.info("Finished Training.")


    def train_one_epoch(self):
        self.log.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']} ...")

        self.lr_scheduler.step()
        for metric in self.metrics:
            metric.clear()
        total_loss = []
        self.model.train()
        for inputs, truths in self.dataloaders["train"]:
            start_time = time.time()
            self.global_step += 1

            inputs = inputs.to(self.device)
            truths = truths.to(self.device)

            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.objective(outputs, truths)
            loss.backward()
            self.optimizer.step()

            for metric in self.metrics:
                metric.update(outputs, truths)

            total_loss.append(loss.item())
            total_time = time.time() - start_time

            if self.global_step % self.report_step_interval == 0:
                self.log.log1(f"step={self.epoch}/{self.global_step}, loss={loss.item():.4f}, time={total_time:.2f}")
                for metric in self.metrics:
                    self.log.log1(metric.report())
                    metric.clear()
        
        average_loss = np.mean(total_loss)
        self.stats["StepLoss"].extend(total_loss)
        self.stats["EpochLoss"].append(average_loss)

        self.log.log(f"==> Finished Epoch {self.epoch}, average_loss={average_loss:.4f}")


    def evaluate(self):
        self.log.log(f"++> Evaluate at epoch {self.epoch} ...")

        self.model.eval()
        for metric in self.metrics:
            metric.clear()

        with torch.no_grad():
            start_time = time.time()
            for inputs, truths in self.dataloaders[self.eval_set]:
                
                inputs = inputs.to(self.device)
                truths = truths.to(self.device)

                outputs = self.model(inputs)

                for metric in self.metrics:
                    metric.update(outputs, truths)

            total_time = time.time() - start_time
            self.log.log1(f"total_time={total_time:.2f}")
            for metric in self.metrics:
                self.log.log1(metric.report())
                metric.clear()

        self.log.log(f"++> Evaluate Finished.")

    def plot_loss(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.set_ylabel("loss")
        ax.set_xlabel("step")
        ax.plot(self.stats["StepLoss"], color='b', alpha=0.3)
        ax2 = ax.twiny()
        ax2.set_xlabel("epoch")
        ax2.plot(self.stats["EpochLoss"], color='r')
        plt.show()


    def save_checkpoint(self):
        """Saves a checkpoint of the modelwork and other variables."""
        with DelayedKeyboardInterrupt():
            model_name = type(self.model).__name__

            state = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_name': model_name,
                'model': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'stats' : self.stats,
            }
            
            ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
            os.makedirs(ckpt_path, exist_ok=True)
            
            file_path = f"{ckpt_path}/{model_name}_ep{self.epoch:04d}.pth.tar"
            torch.save(state, file_path)


    def load_checkpoint(self, checkpoint = None):
        """Loads a modelwork checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the modelwork at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        model_name = type(self.model).__name__
        
        ckpt_path = os.path.join(self.workspace_path, 'checkpoints')
        
        if checkpoint is None:
            # Load most recent checkpoint            
            checkpoint_list = sorted(glob.glob(f'{ckpt_path}/{model_name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                self.log.info("No matching checkpoint found, train from zero.")
                return False
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = f'{ckpt_path}/{model_name}_ep{checkpoint:04d}.pth.tar'
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        checkpoint_dict = torch.load(checkpoint_path)

        #assert model_name == checkpoint_dict['model_name'], 'network is not of correct type.'

        self.epoch = checkpoint_dict['epoch'] + 1
        self.global_step = checkpoint_dict['global_step']
        self.model.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
        self.lr_scheduler.last_epoch = checkpoint_dict['epoch'] 
        self.stats = checkpoint_dict['stats']
        
        self.log.info("Checkpoint Loaded Successfully.")
        return True


