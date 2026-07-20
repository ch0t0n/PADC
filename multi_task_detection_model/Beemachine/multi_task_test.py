import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,7"
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.cuda.amp import autocast

import lightning as L 
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import segmentation_models_pytorch as smp

# ==========================================
# 1. DATASET
# ==========================================
class SpeciesPartDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, image_size, num_classes):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes # Number of segmentation part classes

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Using your custom normalization values
            transforms.Normalize(mean=(0.4925, 0.4475, 0.3490), 
                                 std=(0.2392, 0.2265, 0.2213))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Image and Species Label from CSV
        img_name = self.df.loc[idx, "images"]
        species_label = self.df.loc[idx, "label"]
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # 2. Get corresponding Mask (Parts)
        # Assumes mask is named 'filename_m.png' or 'filename.png'
        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + "_m.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base + ".png")
            
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        
        # Safety: Ensure mask values don't exceed num_classes
        mask_np[mask_np >= self.num_classes] = 0
        
        return img, torch.from_numpy(mask_np), torch.tensor(species_label, dtype=torch.long)

# ==========================================
# 2. MULTI-TASK LIGHTNING MODULE
# ==========================================
class MultiTaskPartModel(L.LightningModule):
    def __init__(self, arch, encoder_name, num_parts, num_species):
        super().__init__()
        self.save_hyperparameters()
        
        aux_params = dict(pooling='avg', dropout=0.3, classes=num_species)

        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name, 
            in_channels=3, 
            classes=num_parts, 
            aux_params=aux_params
        )

        # self.seg_loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def _compute_metrics(self, mask_logits, mask, species_logits, species_label, stage):
        acc = (species_logits.argmax(1) == species_label).float().mean()

        # 🔥 DO NOT compute IoU here
        self.log(
            f"{stage}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
    
    def dice_loss(self, logits, targets, eps=1e-6):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))

        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice.mean()

    def training_step(self, batch, batch_idx):
        img, mask, species_label = batch
        mask_logits, species_logits = self(img)
        
        
        loss_seg = self.dice_loss(mask_logits, mask)
        loss_cls = self.cls_loss_fn(species_logits, species_label)
        total_loss = loss_seg + loss_cls
        
        self._compute_metrics(mask_logits, mask, species_logits, species_label, "train")
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        img, mask, species_label = batch
        mask_logits, species_logits = self(img)
        
        # ✅ LOG VALIDATION CLASSIFICATION ACCURACY
        self._compute_metrics(
            mask_logits, mask,
            species_logits, species_label,
            stage="val"
        )

        preds = mask_logits.argmax(dim=1)
        
        tp, fp, fn, tn = smp.metrics.get_stats(preds, mask, mode="multiclass", num_classes=self.hparams.num_parts)
        
        # Still store for epoch-end aggregation if needed
        self.validation_step_outputs.append({"tp": tp.detach(), "fp": fp.detach(), "fn": fn.detach(), "tn": tn.detach()})
        # return loss_seg + loss_cls

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        tp = torch.cat([x["tp"] for x in self.validation_step_outputs], dim=0)
        fp = torch.cat([x["fp"] for x in self.validation_step_outputs], dim=0)
        fn = torch.cat([x["fn"] for x in self.validation_step_outputs], dim=0)
        tn = torch.cat([x["tn"] for x in self.validation_step_outputs], dim=0)

        # 🔥 Move to CPU explicitly
        tp, fp, fn, tn = tp.cpu(), fp.cpu(), fn.cpu(), tn.cpu()

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

        self.log(
            "val_iou",
            iou,
            prog_bar=True
        )

        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def test_step(self, batch, batch_idx):
        img, mask, species_label = batch
        mask_logits, species_logits = self(img)
        
        # Reuse the metric helper we built earlier
        self._compute_metrics(mask_logits, mask, species_logits, species_label, "test")

    def on_test_epoch_end(self):
        # This is called automatically at the end of the test loop
        print("\n" + "="*30)
        print("TESTING COMPLETE")
        # Lightning will automatically print the logged values in a table format

# ==========================================
# 1. CONFIGURATION
# ==========================================
EPOCHS = 100
BATCH_SIZE = 128
IMAGE_SIZE = 320
CURRENT_DATASET = "Beemachine" # Switch between "BEE", "BIRD", "FISH"
MODEL_NAME = "deeplabv3plus"
ENCODER_NAME = "resnext50_32x4d" 
DATASET_CONFIG = {
    "Beemachine": {"root": r"/home/c/choton/beemachine/datasets/Beemachine_Partwhole_v5/", "num_classes": 4},
    "BIRD": {"root": "path/to/bird", "num_classes": 12},
    "FISH": {"root": "path/to/fish", "num_classes": 10},
}
CFG = DATASET_CONFIG[CURRENT_DATASET]



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
        image_size=IMAGE_SIZE,
        num_classes=CFG["num_classes"]
    )
    val_dataset = SpeciesPartDataset(df=val_df, 
        img_dir=os.path.join(DATA_DIR, "valid", "images"),
        mask_dir=os.path.join(DATA_DIR, "valid", "masks"),
        image_size=IMAGE_SIZE,
        num_classes=CFG["num_classes"]
    )
    test_dataset = SpeciesPartDataset(df=train_df, 
        img_dir=os.path.join(DATA_DIR, "test", "images"),
        mask_dir=os.path.join(DATA_DIR, "test", "masks"),
        image_size=IMAGE_SIZE,
        num_classes=CFG["num_classes"]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

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
        num_species=num_classes # from your LabelEncoder
    )

    # 3. Modern L.Trainer
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        strategy="ddp", # This ensures correct multi-GPU backend initialization
        logger=csv_logger,
        callbacks=[
            ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=1),
            LearningRateMonitor(logging_interval="step")
        ]
    )

    # 4. Run
    trainer.fit(model, train_loader, val_loader)

    # 6. Test
    results = trainer.test(model, dataloaders=test_loader, ckpt_path="best")