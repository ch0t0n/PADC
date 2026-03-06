import os
import ast
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import segmentation_models_pytorch as smp


class PartWholeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, df=None, image_size=224, mask_sfx='.png'):
        self.df = df
        self.mask_sfx = mask_sfx # Mask suffix "_m.png"
        self.image_size = image_size
        self.image_dir = image_dir # os.path.join(data_path, 'Images')
        self.mask_dir = mask_dir # os.path.join(data_path, 'segmentation_masks', 'images')
        self.image_names = list(self.df['filename']) # os.listdir(self.image_dir)
        # self.image_paths = [os.path.join(self.image_dir, p) for p in self.image_names]

        # transform for image
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.PILToTensor(),  # keeps integer values
        ])
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load the image at index idx and convert to tensor
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img_pil = Image.open(img_path)
        img_tens = self.img_transform(img_pil)

        # Load the corresponding mask and convert to tensor
        mask_path = os.path.join(self.mask_dir, img_name)[:-4]+self.mask_sfx
        mask_pil = Image.open(mask_path)
        mask_tens = self.mask_transform(mask_pil)

        cls_label = self.df['standardized_species'][idx]
        return img_tens, mask_tens.squeeze(0), cls_label

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
    MODEL_NAME = "deeplabv3plus" # Available options are: ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan', 'upernet', 'segformer', 'dpt']"
    ENCODER_NAME = "resnext50_32x4d" 
    DEVICE_ID = 1
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 320
    SEED = 42

    # Seeding
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Create Dataset
    DATA_DIR = r'/home/c/choton/beemachine/datasets/Others/fish-vista' # Define the Fish-Vista dataset directory

    # Load the segmentation splits and check the shape
    seg_train_csv = os.path.join(DATA_DIR, r'segmentation_train.csv')
    seg_train_aug_csv = os.path.join(DATA_DIR, r'segmentation_train_aug.csv')
    seg_val_csv = os.path.join(DATA_DIR, r'segmentation_val.csv')
    seg_test_csv = os.path.join(DATA_DIR, r'segmentation_test.csv')
    seg_train_df = pd.read_csv(seg_train_csv)
    seg_train_aug_df = pd.read_csv(seg_train_aug_csv)
    seg_val_df = pd.read_csv(seg_val_csv)
    seg_test_df = pd.read_csv(seg_test_csv)
    print(f'Shape of FishVista segmentation datasets,  train: {seg_train_df.shape}, train_aug: {seg_train_aug_df.shape}, validation: {seg_val_df.shape}, test): {seg_test_df.shape}')
    print(f'Columns of the test dataset:', list(seg_test_df.columns))

    # Class labels of the segmentation split
    train_species = set(list(seg_train_df['standardized_species']))
    val_species = set(list(seg_val_df['standardized_species']))
    test_species = set(list(seg_test_df['standardized_species']))
    check_val = [v for v in val_species if v not in train_species]
    check_test = [v for v in test_species if v not in train_species]
    print(f'The number of classes (species) are, train: {len(train_species)}, val: {len(val_species)}, test: {len(test_species)}')
    print(f'Species in val but not train: {len(check_val)},  species in test but not train: {len(check_test)}')

    # Read the mask labels (traits)
    seg_json_path = os.path.join(DATA_DIR, 'segmentation_masks', 'seg_id_trait_map.json')
    with open(seg_json_path, 'r') as json_file:
        content = json_file.read()
        seg_json = ast.literal_eval(content)
        print('Names of the mask labels (traits):')
        print(seg_json)
    labels = list(seg_json.values())

    imgs_path = os.path.join(DATA_DIR, 'Images')
    masks_path = os.path.join(DATA_DIR, 'segmentation_masks', 'images')
    aug_imgs_path = os.path.join(DATA_DIR, 'train_aug_images')
    aug_masks_path = os.path.join(DATA_DIR, 'train_aug_masks')

    train_dataset = PartWholeDataset(image_dir=imgs_path, mask_dir=masks_path, df=seg_train_df, image_size=IMAGE_SIZE)
    train_aug_dataset = PartWholeDataset(image_dir=aug_imgs_path, mask_dir=aug_masks_path, df=seg_train_aug_df, image_size=IMAGE_SIZE, mask_sfx='_m.png')
    val_dataset = PartWholeDataset(image_dir=imgs_path, mask_dir=masks_path, df=seg_val_df, image_size=IMAGE_SIZE)
    test_dataset = PartWholeDataset(image_dir=imgs_path, mask_dir=masks_path, df=seg_test_df, image_size=IMAGE_SIZE)

    num_classes = len(labels)
    print(f"Training with number of part labels (including background) = {num_classes}")
    print("Part labels:", labels)

    # DataLoaders
    train_loader = DataLoader(train_aug_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    model = CamVidModel(MODEL_NAME, ENCODER_NAME, in_channels=3, out_classes=num_classes, learning_rate=LEARNING_RATE)
    csv_logger = CSVLogger(save_dir=".", name="lightning_logs")

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, devices=[DEVICE_ID], logger=csv_logger)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":
    main()