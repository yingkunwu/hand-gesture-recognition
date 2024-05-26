import os
import torch
import yaml
import argparse
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model.multitasknet import MultiTaskNet
from libs.loss import JointsMSELoss, ClassificationLoss
from libs.metrics import pose_accuracy
from libs.vis import save_debug_images
from libs.load import HandDataModule

# Faster, but less precise
torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)


class MultiTaskModule(LightningModule):
    def __init__(self, data_cfg, image_size, batch_size,
                 lr, lr_step, lr_factor, save_path):
        super().__init__()

        num_joints = data_cfg['num_joints']
        num_classes = data_cfg['num_classes']
        self.names = data_cfg['names']

        self.model = MultiTaskNet(num_joints, num_classes, image_size)

        self.joints_loss = JointsMSELoss(use_target_weight=True)
        self.class_loss = ClassificationLoss()

        self.batch_size = batch_size
        self.lr = lr
        self.lr_step = lr_step
        self.lr_factor = lr_factor

        self.output_path = save_path
        self.save_hyperparameters()

        self.y_pred = []
        self.y_true = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), self.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, batch, batch_idx):
        img, label, target, target_weight, _ = batch

        pred_label, heatmap, attnmap = self.model(img)

        class_loss = self.class_loss(pred_label, label) * 0.001
        joints_loss = self.joints_loss(heatmap, target, target_weight)

        pred_label = torch.argmax(pred_label, dim=1)
        cls_f1score = f1_score(pred_label.detach().cpu().numpy(),
                               label.detach().cpu().numpy(),
                               average='macro')

        _, avg_acc, cnt, pred_joints = pose_accuracy(
            heatmap.detach().cpu().numpy(),
            target.detach().cpu().numpy())

        total_loss = class_loss + joints_loss

        loss = {
            "total_loss": total_loss,
            "class_loss": class_loss.item(),
            "joints_loss": joints_loss.item(),
        }

        return loss, pred_label, cls_f1score, \
            pred_joints, heatmap, avg_acc, cnt, attnmap

    def training_step(self, batch, batch_idx):
        (loss, pred_label, cls_f1score, pred_joints,
         heatmap, avg_acc, cnt, attnmap) = self.forward(batch, batch_idx)
        self.train_count += cnt
        self.train_total_acc += avg_acc * cnt

        log = {}
        for key, value in loss.items():
            log[f"train/{key}"] = value
        log.update({'train/cls_f1score': cls_f1score})
        log.update({'train/pose_acc': self.train_total_acc / self.train_count})

        self.log_dict(
            log,
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size)

        return {"loss": loss["total_loss"], "pred_label": pred_label,
                "pred_joints": pred_joints, "heatmap": heatmap,
                "attnmap": attnmap}

    def validation_step(self, batch, batch_idx):
        (loss, pred_label, cls_f1score, pred_joints,
         heatmap, avg_acc, cnt, attnmap) = self.forward(batch, batch_idx)
        self.val_count += cnt
        self.val_total_acc += avg_acc * cnt

        log = {}
        for key, value in loss.items():
            log[f"val/{key}"] = value
        log.update({'val/cls_f1score': cls_f1score})
        log.update({'val/pose_acc': self.val_total_acc / self.val_count})

        self.log_dict(
            log,
            logger=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size)

        return {"loss": loss["total_loss"], "pred_label": pred_label,
                "pred_joints": pred_joints, "heatmap": heatmap,
                "attnmap": attnmap}

    def test_step(self, batch, batch_idx):
        _, label, _, _, _ = batch
        _, pred_label, _, _, _, _, _, _ = self.forward(batch, batch_idx)

        self.y_pred.extend(pred_label.detach().cpu().numpy().tolist())
        self.y_true.extend(label.detach().cpu().numpy().tolist())

    def on_train_epoch_start(self):
        self.train_count = 0
        self.train_total_acc = 0

    def on_validation_epoch_start(self):
        self.val_count = 0
        self.val_total_acc = 0

    def on_train_batch_end(self, out, batch, batch_idx):
        if batch_idx % 100 == 0:
            img, label, target, _, meta = batch

            pred_label = out["pred_label"]
            pred_joints = out["pred_joints"]
            heatmap = out["heatmap"]
            attnmap = out["attnmap"]
            prefix = '{}_{}'.format(
                os.path.join(self.output_path, 'train'), batch_idx)
            save_debug_images(
                img, prefix, pred_label, label,
                pred_joints*4, heatmap, meta, target, attnmap)

    def on_validation_batch_end(self, out, batch, batch_idx):
        if batch_idx % 100 == 0:
            img, label, target, _, meta = batch

            pred_label = out["pred_label"]
            pred_joints = out["pred_joints"]
            heatmap = out["heatmap"]
            attnmap = out["attnmap"]
            prefix = '{}_{}'.format(
                os.path.join(self.output_path, 'val'), batch_idx)
            save_debug_images(
                img, prefix, pred_label, label,
                pred_joints*4, heatmap, meta, target, attnmap)

    def on_test_epoch_end(self):
        cls_f1score = f1_score(self.y_pred, self.y_true, average='macro')
        print("Test F1 Score: {:.4f}".format(cls_f1score))

        # generate covariance matrix of predicted classes and ground truths
        cm = confusion_matrix(self.y_pred, self.y_true)
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=self.names.keys())
        fig, ax = plt.subplots(figsize=(10, 10))
        cmd.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "confusion_matrix.png"))


def run(args, data_cfg):
    assert args.image_size[0] == args.image_size[1], \
        "Only support square images for now."

    model_name = "{}_{}x{}_{}".format(
        args.backbone, args.image_size[0], args.image_size[1], args.suffix)

    save_path = os.path.join(args.save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    dm = HandDataModule(
        data_cfg,
        args.image_size,
        args.batch_size,
        args.sigma,
        args.num_workers)
    module = MultiTaskModule(
        data_cfg,
        args.image_size,
        args.batch_size,
        args.lr, args.lr_step, args.lr_factor,
        save_path)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(save_path, "weight"),
        filename="best",
        monitor='val/total_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        save_weights_only=True)
    callbacks = [lr_monitor, ckpt_cb]

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=model_name)

    trainer = Trainer(accelerator='gpu',
                      devices=[args.device],
                      precision=32,
                      max_epochs=args.epochs,
                      deterministic=False,
                      num_sanity_val_steps=1,
                      logger=logger,
                      callbacks=callbacks)

    trainer.fit(module, dm)

    # testing
    trainer.test(module, dm, "best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str,
                        default='', help='path to the data config',
                        required=True)
    parser.add_argument('--suffix', type=str,
                        default='', help='suffix of the model name',
                        required=True)
    parser.add_argument('--device', type=int,
                        default=0, help='gpu device to be used')
    parser.add_argument('--backbone', type=str,
                        default='gelans',
                        choices=['resnet18', 'resnet50', 'resnext50',
                                 'gelans', 'gelanl'],
                        help='backbone to be used')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--epochs', type=int,
                        default=50, help='epochs')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--lr_step', nargs='+', type=int,
                        default=[30, 40], help='learning rate step')
    parser.add_argument('--lr_factor', type=float,
                        default=0.1, help='learning rate factor')
    parser.add_argument('--image_size', nargs='+', type=int,
                        default=[192, 192],
                        help='image size (only support square images)')
    parser.add_argument('--sigma', type=int,
                        default=2,
                        help='std of the gaussian distribution heatmap')
    parser.add_argument('--log_dir', type=str,
                        default='logs',
                        help='directory to save the logs')
    parser.add_argument('--save_dir', type=str,
                        default='output',
                        help='directory to save the output')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='number of workers for dataloader')

    args = parser.parse_args()
    print(args)

    with open(args.data_config, "r") as stream:
        try:
            data_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            assert False, exc

    run(args, data_cfg)
