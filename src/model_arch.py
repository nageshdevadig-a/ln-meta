import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class MobileNetUNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Matches Cell 3 exactly
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        self.lossfn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)