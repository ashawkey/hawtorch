# Hawtorch

This is a framework extension for Pytorch, aiming at **code re-use, visualization and neat logging**.

The codes are simple to read, and this package encourages users to read and understand the source codes when they encountered problems.

All you need to do is to **write a pytorch model, a data provider and a main program** to drive them. 

Examples can be found under the corresponding folders.

### Structure

* `trainer`

  A easy-to-modify general `Trainer` class for any model and metrics.

  * auto save, load checkpoints.
  * `tqdm` , `tensorboardX` integration

* `io`

  Simplified input & output API, and a `logger` class for creating logs while training.

  * beautiful and clear logging style.

* `utils`

  * `DelayedKeyboardInterrupt`

    This is used to avoid saving incomplete checkpoints.

  * `EmailSender`

    This can be used to send you an email reporting the reports and logs when the training process is ended.

* `nn`

  Custom Neural Networks, Functionals and Losses as in Pytorch.

* `optim`

  Custom optimizers.

* `vision`

    Simplified API for Image-specific operations.

    * plot image by filenames
    * 3D point cloud / graph 

* `metrics`

    Evaluation metrics for different tasks. 

