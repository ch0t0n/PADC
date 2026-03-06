# ==========================================
# 1. CONFIGURATION
# ==========================================
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 320
DEVICE_IDS = [0,1,2,3,4,5,6,7]  # List of GPU IDs
CURRENT_DATASET = "Beemachine"  # Switch between "BEE", "BIRD", "FISH"
MODEL_NAME = "deeplabv3plus"
ENCODER_NAME = "tu-convnext_nano"
DATASET_CONFIG = {
    "Beemachine": {"root": r"/home/c/choton/beemachine/datasets/Beemachine_Partwhole_v5/", "num_classes": 4},
    "BIRD": {"root": "path/to/bird", "num_classes": 12},
    "FISH": {"root": "path/to/fish", "num_classes": 10},
}
CFG = DATASET_CONFIG[CURRENT_DATASET]

SHAPE_PATH = r'/home/c/choton/beemachine/codes/AG_vision_2026/GT_feature_analysis/Beemachine/predicted_shape_features'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import segmentation_models_pytorch as smp

# ==========================================
# 1. DATASET
# ==========================================
class SpeciesPartDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir,
                 shape_csv_path,
                 image_size, num_classes):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes  # Number of segmentation part classes
        self.scaler = RobustScaler()

        # ---------------------------------
        # Load shape descriptor CSV
        # ---------------------------------
        shape_df = pd.read_csv(shape_csv_path)

        # Merge with main dataframe
        self.df = df.merge(shape_df, on="image", how="inner").reset_index(drop=True)

        # Identify descriptor columns
        self.shape_columns = [
            c for c in self.df.columns
            if c not in ["image", "species", "label"]
        ]
        self.scaler.fit(self.df[self.shape_columns])

        shape_df[self.shape_columns] = shape_df[self.shape_columns].clip(-1e5, 1e5)
        
        self.shape_mean = self.df[self.shape_columns].mean().values
        self.shape_std = self.df[self.shape_columns].std().values + 1e-6

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4925, 0.4475, 0.3490),
                                 std=(0.2392, 0.2265, 0.2213))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Get Image and Species Label
        img_name = row["image"]
        species_label = row["label"]

        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # 2. Get corresponding Mask (Parts)
        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + "_m.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base + ".png")

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        mask_np[mask_np >= self.num_classes] = 0

        # 3. Load precomputed shape features
        shape_feats = row[self.shape_columns].values.astype(np.float32)
        shape_feats = self.scaler.transform(shape_feats.reshape(1, -1)).flatten()
        shape_feats = np.nan_to_num(shape_feats, nan=0.0)

        return (
            img,
            torch.from_numpy(mask_np),
            torch.tensor(species_label, dtype=torch.long),
            torch.tensor(shape_feats, dtype=torch.float32)
        )


# ==========================================
# 2. MULTI-TASK LIGHTNING MODULE
# ==========================================
class MultiTaskPartModel(L.LightningModule):
    def __init__(self, arch, encoder_name, num_parts, num_species, shape_dim=937):
        super().__init__()
        self.save_hyperparameters()
        
        # aux_params = dict(
        #     pooling='avg',
        #     dropout=0.3,
        #     classes=num_species,
        # )

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=3,
            classes=num_parts,
            # aux_params=aux_params
        )

        # Newly added

        # Classification head (visual + shape)
        self.classifier = nn.Sequential(
            nn.Linear(shape_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_species)
        )

        self.seg_loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, smooth=1.0)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        self.validation_step_outputs = []

    def forward(self, x, shape_feats):
        # Segmentation output
        mask_logits = self.model(x)
        # fused = torch.cat([shape_feats], dim=1)
        species_logits = self.classifier(shape_feats)
        return mask_logits, species_logits

    def _compute_metrics(self, mask_logits, mask, species_logits, species_label, stage):
        # 1. Classification Accuracy
        acc = (species_logits.argmax(1) == species_label).float().mean()

        # 2. Segmentation IoU (Macro)
        preds = mask_logits.argmax(dim=1)
        ious = []
        for cls in range(self.hparams.num_parts):
            pred_inds = (preds == cls)
            target_inds = (mask == cls)
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            if union == 0:
                ious.append(1.0)
            else:
                ious.append(intersection / union)

        iou = sum(ious) / len(ious)

        self.log(f"{stage}_acc", acc,
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        self.log(f"{stage}_iou", iou,
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        img, mask, species_label, shape_feats = batch

        mask_logits, species_logits = self(img, shape_feats)

        loss_seg = self.seg_loss_fn(mask_logits, mask)
        loss_cls = self.cls_loss_fn(species_logits, species_label)
        total_loss = loss_seg + loss_cls

        self._compute_metrics(mask_logits, mask, species_logits, species_label, "train")

        self.log("train_loss", total_loss,
                 on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        img, mask, species_label, shape_feats = batch

        mask_logits, species_logits = self(img, shape_feats)

        loss_seg = self.seg_loss_fn(mask_logits, mask)
        loss_cls = self.cls_loss_fn(species_logits, species_label)

        self._compute_metrics(mask_logits, mask, species_logits, species_label, "val")

        return loss_seg + loss_cls

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def test_step(self, batch, batch_idx):
        img, mask, species_label, shape_feats = batch

        mask_logits, species_logits = self(img, shape_feats)

        self._compute_metrics(mask_logits, mask, species_logits, species_label, "test")

    def on_test_epoch_end(self):
        print("\n" + "="*30)
        print("TESTING COMPLETE")


# ==========================================
# 3. UPDATED TRAINER EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup Data
    DATA_DIR = r"/home/c/choton/beemachine/datasets/Beemachine_Partwhole_v5/"
    train_datapath = os.path.join(DATA_DIR, 'train_aug_labels.csv')
    val_datapath = os.path.join(DATA_DIR, 'val_labels.csv')
    test_datapath = os.path.join(DATA_DIR, 'test_labels.csv')

    train_df = pd.read_csv(train_datapath)
    val_df = pd.read_csv(val_datapath)
    test_df = pd.read_csv(test_datapath)

    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["species"])
    val_df["label"] = le.transform(val_df["species"])
    test_df["label"] = le.transform(test_df["species"])
    num_classes = len(le.classes_)
    print(f"Total images, Train: {len(train_df['label'])}, Validation: {len(val_df['label'])}, Test: {len(test_df['label'])}")
    print(f"Total classes: {num_classes}")

    train_dataset = SpeciesPartDataset(df=train_df,
        img_dir=os.path.join(DATA_DIR, "train", "aug_images"),
        mask_dir=os.path.join(DATA_DIR, "train", "aug_masks"),
        shape_csv_path=os.path.join(SHAPE_PATH, "beemachine_partwhole_v5_train.csv"),
        image_size=IMAGE_SIZE,
        num_classes=CFG["num_classes"]
    )
    val_dataset = SpeciesPartDataset(df=val_df,
        img_dir=os.path.join(DATA_DIR, "valid", "images"),
        mask_dir=os.path.join(DATA_DIR, "valid", "masks"),
        shape_csv_path=os.path.join(SHAPE_PATH, "beemachine_partwhole_v5_val.csv"),
        image_size=IMAGE_SIZE,
        num_classes=CFG["num_classes"]
    )
    test_dataset = SpeciesPartDataset(df=test_df,  # Fixed typo: was train_df
        img_dir=os.path.join(DATA_DIR, "test", "images"),
        mask_dir=os.path.join(DATA_DIR, "test", "masks"),
        shape_csv_path=os.path.join(SHAPE_PATH, "beemachine_partwhole_v5_test.csv"),
        image_size=IMAGE_SIZE,
        num_classes=CFG["num_classes"]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Images in dataset, train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    csv_logger = CSVLogger(
        save_dir="logs/",
        name="beemachine_v5_experiment"
    )

    # 2. Setup Model
    model = MultiTaskPartModel(
        arch=MODEL_NAME,
        encoder_name=ENCODER_NAME,
        num_parts=CFG["num_classes"],
        num_species=num_classes  # from your LabelEncoder
    )

    # 3. Modern L.Trainer
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=DEVICE_IDS,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=1, filename='best-part-model-{epoch:02d}-{val_iou:.2f}'),
            LearningRateMonitor(logging_interval="step")
        ],
        gradient_clip_val=1.0,          # ← very helpful with mixed precision
        gradient_clip_algorithm="value",
    )

    # 4. Run
    trainer.fit(model, train_loader, val_loader)

    # 6. Test
    results = trainer.test(model, dataloaders=test_loader, ckpt_path="best")