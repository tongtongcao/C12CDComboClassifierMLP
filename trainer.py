import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class comboClassifierMLP(pl.LightningModule):

    def __init__(self, input_dim=39, hidden_dim=64, num_layers=3, lr=1e-3, dropout=0.1):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr

        # Backbone
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Binary classification
        self.classifier = nn.Linear(hidden_dim, 1)

        # Output is a probability after sigmoid
        self.criterion = nn.BCELoss()

    """
        Parameters
        ----------
        x_cont : torch.Tensor
        Shape: [batch_size, 39]
        
        Returns
        -------
        probability : torch.Tensor
            Shape: [batch_size, 1]
            Probability that the candidate is valid.
    """
    def forward(self, x_cont):
        h = self.backbone(x_cont)
        logits = self.classifier(h)
        probability = torch.sigmoid(logits)

        return probability

    def training_step(self, batch, batch_idx):
        x_cont, labels = batch
        labels = labels.float().view(-1, 1)
        probability = self(x_cont)
        loss = self.criterion(probability, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_cont, labels = batch
        labels = labels.float().view(-1, 1)
        probability = self(x_cont)
        loss = self.criterion(probability, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class LossTracker(Callback):
    """
    PyTorch Lightning callback to track
    training and validation losses.
    """

    def __init__(self):
        super().__init__()

        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())