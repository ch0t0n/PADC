import os
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import segmentation_models_pytorch as smp

def read_part_labels(path):
    with open(path) as f:
        labels = [l.strip() for l in f if l.strip()]
        return labels

class PartWholeDataset(Dataset):
    def __init__(self, root, images_dir="aug_images", masks_dir="aug_masks", image_size=320):

        self.images_dir = os.path.join(root, images_dir)
        self.masks_dir = os.path.join(root, masks_dir)
        self.image_size = image_size

        # -------- Labels --------
        self.labels = ["background"] + read_part_labels(
            os.path.join(root, "part_labels.txt")
        )
        self.num_parts = len(self.labels)

        # class folders
        self.classes = sorted(os.listdir(self.images_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # collect image paths WITH class
        self.samples = []
        for cls in self.classes:
            cls_img_dir = os.path.join(self.images_dir, cls)
            for fname in os.listdir(cls_img_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(cls_img_dir, fname)
                    mask_path = os.path.join(self.masks_dir, cls, fname.replace(".jpg", "_m.png"))
                    self.samples.append((img_path, mask_path, self.class_to_idx[cls]))

        self.samples = sorted(self.samples)

        # image transform
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, class_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size),
                           resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img, mask, class_id

class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, learning_rate, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask, _ = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    MODEL_NAME = "linknet" # Available options are: ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan', 'upernet', 'segformer', 'dpt']"
    ENCODER_NAME = "resnext50_32x4d" 
    DEVICE_ID = 3
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    IMAGE_SIZE = 320
    SEED = 42

    # Create Dataset
    dataset_path = r'/home/c/choton/beemachine/datasets/Others/CUB_200_2011/'
    full_dataset = PartWholeDataset(root=dataset_path, images_dir="aug_images", masks_dir="aug_masks", image_size=IMAGE_SIZE)
    torch.manual_seed(SEED)

    # Train / Val / Test Split (75:15:10)
    n_total = len(full_dataset)
    n_train = int(0.75 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    num_classes = len(full_dataset.classes)
    print("Number of species:", num_classes)
    print("species names:", full_dataset.classes)
    print("Number of parts", full_dataset.num_parts)
    print("Part labels:", full_dataset.labels)


    model = CamVidModel(MODEL_NAME, ENCODER_NAME, in_channels=3, out_classes=full_dataset.num_parts, learning_rate=LEARNING_RATE)
    csv_logger = CSVLogger(save_dir=".", name="lightning_logs")

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, devices=[DEVICE_ID], logger=csv_logger)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":
    main()