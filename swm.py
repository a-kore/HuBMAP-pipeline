import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from argparse import ArgumentParser
from torchvision.utils import make_grid
from utils import DiceLoss
import segmentation_models_pytorch as smp


class U2Net(pl.LightningModule):
    def __init__(self,
                 in_ch=3,
                 out_ch=1,
                 lr=1.0e-3,
                 weight_decay=3.0e-6,
                 ratio=0.9,
                 **kwargs,
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = smp.Unet(
            encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )

        self.dice_loss = DiceLoss()
        self.ratio = ratio

        self.sample_imgs = torch.zeros(1)
        self.sample_masks = torch.zeros(1)
        self.rand_mask = torch.randint(16, (8,))

    def forward(self, x):
        x_hat = torch.sigmoid(self.model(x))
        return x_hat

    def training_step(self, batch, batch_idx):
        img, mask = batch
        mask_hat = self.forward(img)

        log_loss = F.binary_cross_entropy(mask_hat, mask)
        dice_loss = self.dice_loss(mask_hat, mask)
        train_loss = log_loss+dice_loss

        self.log('train_loss', log_loss, on_epoch=True, on_step=True)
        self.log('train_diceloss', dice_loss, on_epoch=True, on_step=True)
        self.log('train_dicemetric', 1-dice_loss, on_epoch=True, on_step=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        mask_hat = self.forward(img)

        log_loss = F.binary_cross_entropy(mask_hat, mask)
        dice_loss = self.dice_loss(mask_hat, mask)
        val_loss = log_loss+dice_loss

        self.log('val_loss', val_loss, on_epoch=True, on_step=False)
        self.log('val_diceloss', dice_loss, on_epoch=True, on_step=False)
        self.log('val_dicemetric', 1-dice_loss, on_epoch=True, on_step=False)

        self.sample_imgs = img[self.rand_mask]
        self.sample_masks = torch.cat([mask[self.rand_mask],
                                       mask_hat[self.rand_mask]], dim=0)

    def on_validation_epoch_end(self) -> None:
        mask_grid = make_grid(self.sample_masks, 8)
        img_grid = make_grid(self.sample_imgs, 8)
        self.logger.experiment.add_image('gt and pred masks', mask_grid, self.current_epoch)
        self.logger.experiment.add_image('images', img_grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--in_ch", type=int, default=3)
        parser.add_argument("--out_ch", type=int, default=1)
        parser.add_argument("--ratio", type=float, default=0.9)

        parser.add_argument("--lr", type=float, default=1.0e-3)
        parser.add_argument("--weight_decay", type=float, default=3e-6)

        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--image_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=-1)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--pin_memory", action='store_true')
        parser.add_argument("--drop_last", action='store_false')
        parser.add_argument("--split", type=float, default=0.8)
        return parser


def cli_main(args=None):
    from datamodules import HuBMAPDataModule

    parser = ArgumentParser()

    parser.add_argument("--logdir", default="logs", type=str)
    parser.add_argument("--name", default="unet34", type=str)
    parser.add_argument("--chkpt_monitor", default="val_dicemetric", type=str)
    parser.add_argument("--mode", default="max", type=str)
    script_args, _ = parser.parse_known_args(args)

    dm_cls = HuBMAPDataModule

    logger = TensorBoardLogger(script_args.logdir, name=script_args.name)
    checkpoint = ModelCheckpoint(monitor=script_args.chkpt_monitor, mode='max',
                                 filename=script_args.name + '-{epoch:03d}-{' + script_args.chkpt_monitor + ':.5f}')
    parser = U2Net.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)

    model = U2Net(**vars(args))
    # model = U2Net.load_from_checkpoint('./logs/u2net_256/version_0/checkpoints/u2net_256-epoch=305-val_dicemetric=0.96.ckpt')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint])
    trainer.fit(model, datamodule=dm)
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
